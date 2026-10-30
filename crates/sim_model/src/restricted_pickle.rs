use crate::parser_limits::{PARSER_DECODED_ALLOCATION_MULTIPLIER, ParserLimitError, ParserLimits};
use sim_tensor::CancellationToken;
use std::collections::BTreeMap;

pub const RESTRICTED_PICKLE_ALLOWLIST_VERSION: u32 = 1;
pub const RESTRICTED_PICKLE_DECODED_ALLOCATION_MULTIPLIER: u64 =
    PARSER_DECODED_ALLOCATION_MULTIPLIER;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllowedPickleTarget {
    pub target: &'static str,
    pub global: bool,
    pub reduce: bool,
    pub build: bool,
}

pub const ALLOWED_PICKLE_TARGETS: &[AllowedPickleTarget] = &[
    allowed("collections.OrderedDict", true, true, false),
    allowed("collections.Counter", true, true, false),
    allowed("torch.Size", true, true, false),
    allowed("torch.Tensor", true, false, false),
    allowed("torch.bfloat16", true, false, false),
    allowed("torch.bool", true, false, false),
    allowed("torch.complex128", true, false, false),
    allowed("torch.complex64", true, false, false),
    allowed("torch.device", true, true, false),
    allowed("torch.float16", true, false, false),
    allowed("torch.float32", true, false, false),
    allowed("torch.float64", true, false, false),
    allowed("torch.float8_e4m3fn", true, false, false),
    allowed("torch.float8_e5m2", true, false, false),
    allowed("torch.int16", true, false, false),
    allowed("torch.int32", true, false, false),
    allowed("torch.int64", true, false, false),
    allowed("torch.int8", true, false, false),
    allowed("torch.per_channel_affine", true, false, false),
    allowed("torch.per_channel_affine_float_qparams", true, false, false),
    allowed("torch.per_channel_symmetric", true, false, false),
    allowed("torch.per_tensor_affine", true, false, false),
    allowed("torch.per_tensor_symmetric", true, false, false),
    allowed("torch.qint32", true, false, false),
    allowed("torch.qint8", true, false, false),
    allowed("torch.quint8", true, false, false),
    allowed("torch.uint16", true, false, false),
    allowed("torch.uint32", true, false, false),
    allowed("torch.uint64", true, false, false),
    allowed("torch.uint8", true, false, false),
    allowed("torch.BFloat16Storage", true, false, false),
    allowed("torch.BoolStorage", true, false, false),
    allowed("torch.ByteStorage", true, false, false),
    allowed("torch.CharStorage", true, false, false),
    allowed("torch.ComplexDoubleStorage", true, false, false),
    allowed("torch.ComplexFloatStorage", true, false, false),
    allowed("torch.DoubleStorage", true, false, false),
    allowed("torch.FloatStorage", true, false, false),
    allowed("torch.HalfStorage", true, false, false),
    allowed("torch.IntStorage", true, false, false),
    allowed("torch.LongStorage", true, false, false),
    allowed("torch.QInt32Storage", true, false, false),
    allowed("torch.QInt8Storage", true, false, false),
    allowed("torch.QUInt8Storage", true, false, false),
    allowed("torch.ShortStorage", true, false, false),
    allowed("torch.UntypedStorage", true, false, false),
    allowed("torch.nn.parameter.Parameter", true, true, true),
    allowed("torch._utils._rebuild_tensor", true, true, false),
    allowed("torch._utils._rebuild_tensor_v2", true, true, false),
    allowed("torch._utils._rebuild_tensor_v3", true, true, false),
    allowed("torch._utils._rebuild_parameter", true, true, false),
    allowed(
        "torch._utils._rebuild_parameter_with_state",
        true,
        true,
        false,
    ),
    allowed("torch._utils._rebuild_qtensor", true, true, false),
    allowed("torch._utils._rebuild_sparse_tensor", true, true, false),
    allowed(
        "torch._utils._rebuild_meta_tensor_no_storage",
        true,
        true,
        false,
    ),
    allowed("torch._tensor._rebuild_from_type_v2", true, true, false),
    allowed("torch.serialization._get_layout", true, true, false),
    allowed("builtins.bytearray", true, true, false),
    allowed("builtins.set", true, true, false),
    allowed("builtins.complex", true, true, false),
    allowed("_codecs.encode", true, true, false),
    allowed("numpy.core.multiarray.scalar", true, true, false),
    allowed("numpy._core.multiarray.scalar", true, true, false),
    allowed("numpy.dtype", true, true, true),
    allowed("numpy.dtypes.Float64DType", true, true, true),
    allowed(
        "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint",
        true,
        true,
        true,
    ),
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafeGlobalsAdmission {
    targets: Vec<&'static str>,
}

impl SafeGlobalsAdmission {
    pub fn targets(&self) -> &[&'static str] {
        &self.targets
    }
}

pub fn add_safe_globals_exact_native(
    requested_targets: &[&str],
    cancellation: &CancellationToken,
) -> Result<SafeGlobalsAdmission, RestrictedPickleError> {
    if requested_targets.len() > ALLOWED_PICKLE_TARGETS.len() {
        return Err(RestrictedPickleError::SafeGlobalRequestTooLarge {
            requested: requested_targets.len(),
            maximum: ALLOWED_PICKLE_TARGETS.len(),
        });
    }
    let mut targets = Vec::new();
    targets
        .try_reserve_exact(requested_targets.len())
        .map_err(|_| RestrictedPickleError::AllocationFailed {
            requested: requested_targets.len(),
        })?;
    for (index, requested) in requested_targets.iter().enumerate() {
        if index % 64 == 0 && cancellation.check().is_err() {
            return Err(RestrictedPickleError::Cancelled);
        }
        let Some(allowed) = ALLOWED_PICKLE_TARGETS
            .iter()
            .find(|entry| entry.global && entry.target == *requested)
        else {
            return Err(RestrictedPickleError::ForbiddenTarget {
                target: (*requested).to_owned(),
                operation: "SAFE_GLOBALS",
            });
        };
        if !targets.contains(&allowed.target) {
            targets.push(allowed.target);
        }
    }
    targets.sort_unstable();
    if cancellation.check().is_err() {
        return Err(RestrictedPickleError::Cancelled);
    }
    Ok(SafeGlobalsAdmission { targets })
}

const fn allowed(
    target: &'static str,
    global: bool,
    reduce: bool,
    build: bool,
) -> AllowedPickleTarget {
    AllowedPickleTarget {
        target,
        global,
        reduce,
        build,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PickleValue {
    None,
    Boolean(bool),
    Integer(i128),
    FloatBits(u64),
    Bytes(Vec<u8>),
    String(String),
    List(Vec<PickleValue>),
    Tuple(Vec<PickleValue>),
    Dictionary(Vec<(PickleValue, PickleValue)>),
    Set(Vec<PickleValue>),
    Global(String),
    Persistent(Box<PickleValue>),
    Reduced {
        target: String,
        arguments: Box<PickleValue>,
        state: Option<Box<PickleValue>>,
    },
}

impl PickleValue {
    pub fn reduction_target(&self) -> Option<&str> {
        match self {
            Self::Global(target) | Self::Reduced { target, .. } => Some(target),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum RestrictedPickleError {
    #[error("restricted pickle parsing was cancelled")]
    Cancelled,
    #[error(transparent)]
    Limit(#[from] ParserLimitError),
    #[error("pickle input is truncated at byte {offset}")]
    Truncated { offset: usize },
    #[error("pickle opcode 0x{opcode:02x} is not permitted at byte {offset}")]
    ForbiddenOpcode { opcode: u8, offset: usize },
    #[error("pickle opcode 0x{opcode:02x} is unknown at byte {offset}")]
    UnknownOpcode { opcode: u8, offset: usize },
    #[error("pickle target {target:?} is not allowlisted for {operation}")]
    ForbiddenTarget {
        target: String,
        operation: &'static str,
    },
    #[error("pickle stack is invalid: {0}")]
    InvalidStack(&'static str),
    #[error("pickle memo reference {0} does not exist")]
    MissingMemo(u32),
    #[error("pickle scalar is invalid: {0}")]
    InvalidScalar(String),
    #[error("pickle string is not valid UTF-8")]
    InvalidUtf8,
    #[error("pickle allocation of {requested} bytes failed")]
    AllocationFailed { requested: usize },
    #[error("safe-global request has {requested} entries, exceeding {maximum}")]
    SafeGlobalRequestTooLarge { requested: usize, maximum: usize },
    #[error("pickle ended without STOP")]
    MissingStop,
}

pub fn parse_restricted_pickle(
    bytes: &[u8],
    limits: &ParserLimits,
) -> Result<PickleValue, RestrictedPickleError> {
    parse_restricted_pickle_inner(bytes, limits, None)
}

pub fn parse_restricted_pickle_cancellable(
    bytes: &[u8],
    limits: &ParserLimits,
    cancellation: &CancellationToken,
) -> Result<PickleValue, RestrictedPickleError> {
    parse_restricted_pickle_inner(bytes, limits, Some(cancellation))
}

fn parse_restricted_pickle_inner(
    bytes: &[u8],
    limits: &ParserLimits,
    cancellation: Option<&CancellationToken>,
) -> Result<PickleValue, RestrictedPickleError> {
    limits.validate()?;
    limits.check(
        "pickle bytes",
        u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        limits.manifest_bytes,
    )?;
    PickleParser::new(bytes, limits, cancellation).parse()
}

struct PickleParser<'a> {
    bytes: &'a [u8],
    cursor: usize,
    frame_end: Option<usize>,
    limits: &'a ParserLimits,
    stack: Vec<PickleValue>,
    marks: Vec<usize>,
    memo: BTreeMap<u32, PickleValue>,
    next_memo: u32,
    value_count: u64,
    allocation_bytes: u64,
    cancellation: Option<&'a CancellationToken>,
}

impl<'a> PickleParser<'a> {
    fn new(
        bytes: &'a [u8],
        limits: &'a ParserLimits,
        cancellation: Option<&'a CancellationToken>,
    ) -> Self {
        Self {
            bytes,
            cursor: 0,
            frame_end: None,
            limits,
            stack: Vec::new(),
            marks: Vec::new(),
            memo: BTreeMap::new(),
            next_memo: 0,
            value_count: 0,
            allocation_bytes: 0,
            cancellation,
        }
    }

    fn parse(mut self) -> Result<PickleValue, RestrictedPickleError> {
        while self.cursor < self.bytes.len() {
            if self
                .cancellation
                .is_some_and(CancellationToken::is_cancelled)
            {
                return Err(RestrictedPickleError::Cancelled);
            }
            self.finish_frame_if_needed()?;
            let offset = self.cursor;
            let opcode = self.byte()?;
            match opcode {
                b'.' => return self.finish(),
                b'(' => self.push_mark()?,
                b'0' => {
                    self.pop()?;
                }
                b'1' => {
                    let mark = self.pop_mark()?;
                    self.stack.truncate(mark);
                }
                b'2' => {
                    self.duplicate_top()?;
                }
                b'N' => self.push(PickleValue::None)?,
                0x88 => self.push(PickleValue::Boolean(true))?,
                0x89 => self.push(PickleValue::Boolean(false))?,
                b'I' => {
                    let value = self.parse_text_integer()?;
                    self.push(value)?;
                }
                b'J' => {
                    let value = i32::from_le_bytes(self.array::<4>()?);
                    self.push(PickleValue::Integer(i128::from(value)))?;
                }
                b'K' => {
                    let value = self.byte()?;
                    self.push(PickleValue::Integer(i128::from(value)))?;
                }
                b'M' => {
                    let value = u16::from_le_bytes(self.array::<2>()?);
                    self.push(PickleValue::Integer(i128::from(value)))?;
                }
                b'L' => {
                    let value = self.parse_text_long()?;
                    self.push(value)?;
                }
                0x8a => {
                    let length = usize::from(self.byte()?);
                    let value = self.parse_binary_long(length)?;
                    self.push(value)?;
                }
                0x8b => {
                    let length = self.length_u32()?;
                    let value = self.parse_binary_long(length)?;
                    self.push(value)?;
                }
                b'F' => {
                    let line = self.line()?;
                    let text = std::str::from_utf8(line)
                        .map_err(|_| RestrictedPickleError::InvalidUtf8)?;
                    let value = text
                        .parse::<f64>()
                        .map_err(|error| RestrictedPickleError::InvalidScalar(error.to_string()))?;
                    self.push(PickleValue::FloatBits(value.to_bits()))?;
                }
                b'G' => {
                    let value = f64::from_be_bytes(self.array::<8>()?);
                    self.push(PickleValue::FloatBits(value.to_bits()))?;
                }
                b'S' => {
                    let line = self.line()?;
                    self.preflight_allocation(0, u64::try_from(line.len()).unwrap_or(u64::MAX))?;
                    let value = parse_quoted_string(line, self.limits.maximum_name_bytes)?;
                    self.charge_payload(value.len())?;
                    self.push(PickleValue::String(value))?;
                }
                b'V' => {
                    let line = self.line()?;
                    if u64::try_from(line.len()).unwrap_or(u64::MAX)
                        > self.limits.maximum_name_bytes
                    {
                        return Err(ParserLimitError::Exceeded {
                            kind: "pickle unicode bytes",
                            actual: u64::try_from(line.len()).unwrap_or(u64::MAX),
                            maximum: self.limits.maximum_name_bytes,
                        }
                        .into());
                    }
                    self.preflight_allocation(0, u64::try_from(line.len()).unwrap_or(u64::MAX))?;
                    let value = decode_raw_unicode_escape(line)?;
                    self.check_name_length(value.len())?;
                    self.charge_payload(value.len())?;
                    self.push(PickleValue::String(value))?;
                }
                b'T' => {
                    let length = self.length_u32()?;
                    let value = self.utf8_string(length)?;
                    self.push(PickleValue::String(value))?;
                }
                b'U' => {
                    let length = usize::from(self.byte()?);
                    let value = self.utf8_string(length)?;
                    self.push(PickleValue::String(value))?;
                }
                b'X' => {
                    let length = self.length_u32()?;
                    let value = self.utf8_string(length)?;
                    self.push(PickleValue::String(value))?;
                }
                0x8c => {
                    let length = usize::from(self.byte()?);
                    let value = self.utf8_string(length)?;
                    self.push(PickleValue::String(value))?;
                }
                0x8d => {
                    let length = self.length_u64()?;
                    let value = self.utf8_string(length)?;
                    self.push(PickleValue::String(value))?;
                }
                b'B' => {
                    let length = self.length_u32()?;
                    let value = self.owned_bytes(length)?;
                    self.push(PickleValue::Bytes(value))?;
                }
                b'C' => {
                    let length = usize::from(self.byte()?);
                    let value = self.owned_bytes(length)?;
                    self.push(PickleValue::Bytes(value))?;
                }
                0x8e | 0x96 => {
                    let length = self.length_u64()?;
                    let value = self.owned_bytes(length)?;
                    self.push(PickleValue::Bytes(value))?;
                }
                b')' => self.push(PickleValue::Tuple(Vec::new()))?,
                b't' => {
                    let values = self.take_mark_values(1)?;
                    self.push(PickleValue::Tuple(values))?;
                }
                0x85 => self.tuple_fixed(1)?,
                0x86 => self.tuple_fixed(2)?,
                0x87 => self.tuple_fixed(3)?,
                b']' => self.push(PickleValue::List(Vec::new()))?,
                b'l' => {
                    let values = self.take_mark_values(1)?;
                    self.push(PickleValue::List(values))?;
                }
                b'a' => {
                    let value = self.pop()?;
                    self.append_list(value)?;
                }
                b'e' => {
                    let values = self.take_mark_values(2)?;
                    self.extend_list(values)?;
                }
                b'}' => self.push(PickleValue::Dictionary(Vec::new()))?,
                b'd' => {
                    let values = self.take_mark_values(2)?;
                    let entries = pair_values(values)?;
                    self.push(PickleValue::Dictionary(entries))?;
                }
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    self.insert_dictionary(key, value)?;
                }
                b'u' => {
                    let values = self.take_mark_values(2)?;
                    let entries = pair_values(values)?;
                    self.extend_dictionary(entries)?;
                }
                0x8f => self.push(PickleValue::Set(Vec::new()))?,
                0x90 => {
                    let values = self.take_mark_values(2)?;
                    self.extend_set(values)?;
                }
                0x91 => {
                    let values = self.take_mark_values(1)?;
                    self.push(PickleValue::Set(values))?;
                }
                b'p' => {
                    let index = parse_u32_line(self.line()?)?;
                    self.memoize(index)?;
                }
                b'q' => {
                    let index = u32::from(self.byte()?);
                    self.memoize(index)?;
                }
                b'r' => {
                    let index = u32::from_le_bytes(self.array::<4>()?);
                    self.memoize(index)?;
                }
                0x94 => {
                    let index = self.next_memo;
                    self.next_memo = self.next_memo.checked_add(1).ok_or_else(|| {
                        RestrictedPickleError::InvalidScalar("memo index overflow".to_owned())
                    })?;
                    self.memoize(index)?;
                }
                b'g' => {
                    let index = parse_u32_line(self.line()?)?;
                    self.push_memo(index)?;
                }
                b'h' => {
                    let index = u32::from(self.byte()?);
                    self.push_memo(index)?;
                }
                b'j' => {
                    let index = u32::from_le_bytes(self.array::<4>()?);
                    self.push_memo(index)?;
                }
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    let target_length = module
                        .len()
                        .checked_add(name.len())
                        .and_then(|value| value.checked_add(1))
                        .ok_or_else(|| {
                            RestrictedPickleError::InvalidScalar(
                                "target length overflow".to_owned(),
                            )
                        })?;
                    self.preflight_allocation(0, u64::try_from(target_length).unwrap_or(u64::MAX))?;
                    let target = join_global(module, name, self.limits.maximum_name_bytes)?;
                    validate_target(&target, TargetOperation::Global)?;
                    self.charge_payload(target.len())?;
                    self.push(PickleValue::Global(target))?;
                }
                0x93 => {
                    let name = value_string(self.pop()?)?;
                    let module = value_string(self.pop()?)?;
                    let target_length = module
                        .len()
                        .checked_add(name.len())
                        .and_then(|value| value.checked_add(1))
                        .ok_or_else(|| {
                            RestrictedPickleError::InvalidScalar(
                                "target length overflow".to_owned(),
                            )
                        })?;
                    self.check_name_length(target_length)?;
                    self.preflight_allocation(0, u64::try_from(target_length).unwrap_or(u64::MAX))?;
                    let target = format!("{module}.{name}");
                    validate_target(&target, TargetOperation::Global)?;
                    self.charge_payload(target.len())?;
                    self.push(PickleValue::Global(target))?;
                }
                b'R' => self.reduce()?,
                b'b' => self.build()?,
                b'P' => {
                    let line = self.line()?;
                    self.check_name_length(line.len())?;
                    self.charge_allocation(2, u64::try_from(line.len()).unwrap_or(u64::MAX))?;
                    let value = PickleValue::String(
                        std::str::from_utf8(line)
                            .map_err(|_| RestrictedPickleError::InvalidUtf8)?
                            .to_owned(),
                    );
                    self.push_precharged(PickleValue::Persistent(Box::new(value)))?;
                }
                b'Q' => {
                    let value = self.pop()?;
                    self.charge_allocation(1, 0)?;
                    self.push_precharged(PickleValue::Persistent(Box::new(value)))?;
                }
                0x80 => {
                    let version = self.byte()?;
                    if version > 5 {
                        return Err(RestrictedPickleError::InvalidScalar(format!(
                            "unsupported pickle protocol {version}"
                        )));
                    }
                }
                0x95 => {
                    if self.frame_end.is_some() {
                        return Err(RestrictedPickleError::InvalidScalar(
                            "nested pickle FRAME opcode".to_owned(),
                        ));
                    }
                    let length = u64::from_le_bytes(self.array::<8>()?);
                    if length == 0 {
                        return Err(RestrictedPickleError::InvalidScalar(
                            "pickle FRAME length must be non-zero".to_owned(),
                        ));
                    }
                    self.limits
                        .check("pickle frame bytes", length, self.limits.manifest_bytes)?;
                    let frame_length = usize::try_from(length).map_err(|_| {
                        RestrictedPickleError::InvalidScalar(
                            "pickle FRAME length overflow".to_owned(),
                        )
                    })?;
                    let frame_end = self.cursor.checked_add(frame_length).ok_or_else(|| {
                        RestrictedPickleError::InvalidScalar("pickle FRAME end overflow".to_owned())
                    })?;
                    if frame_end > self.bytes.len() {
                        return Err(RestrictedPickleError::Truncated {
                            offset: self.cursor,
                        });
                    }
                    self.frame_end = Some(frame_end);
                }
                0x81 | 0x82 | 0x83 | 0x84 | 0x92 | 0x97 | 0x98 | b'i' | b'o' => {
                    return Err(RestrictedPickleError::ForbiddenOpcode { opcode, offset });
                }
                _ => return Err(RestrictedPickleError::UnknownOpcode { opcode, offset }),
            }
        }
        Err(RestrictedPickleError::MissingStop)
    }

    fn finish(mut self) -> Result<PickleValue, RestrictedPickleError> {
        if self
            .frame_end
            .is_some_and(|frame_end| self.cursor != frame_end)
        {
            return Err(RestrictedPickleError::InvalidScalar(
                "pickle STOP precedes the FRAME boundary".to_owned(),
            ));
        }
        if self.cursor != self.bytes.len() {
            return Err(RestrictedPickleError::InvalidScalar(
                "pickle has trailing bytes after STOP".to_owned(),
            ));
        }
        if !self.marks.is_empty() || self.stack.len() != 1 {
            return Err(RestrictedPickleError::InvalidStack(
                "STOP requires one value and no open MARK",
            ));
        }
        self.pop()
    }

    fn push(&mut self, value: PickleValue) -> Result<(), RestrictedPickleError> {
        value_metrics(&value, 1, self.limits.maximum_depth)?;
        self.charge_allocation(1, 0)?;
        self.push_precharged(value)
    }

    fn push_precharged(&mut self, value: PickleValue) -> Result<(), RestrictedPickleError> {
        self.stack
            .try_reserve(1)
            .map_err(|_| RestrictedPickleError::AllocationFailed {
                requested: std::mem::size_of::<PickleValue>(),
            })?;
        self.stack.push(value);
        Ok(())
    }

    fn push_mark(&mut self) -> Result<(), RestrictedPickleError> {
        let next_depth = self.marks.len().saturating_add(1);
        self.limits.check(
            "pickle mark depth",
            u64::try_from(next_depth).unwrap_or(u64::MAX),
            u64::from(self.limits.maximum_depth),
        )?;
        self.preflight_allocation(
            0,
            u64::try_from(std::mem::size_of::<usize>()).unwrap_or(u64::MAX),
        )?;
        self.marks
            .try_reserve(1)
            .map_err(|_| RestrictedPickleError::AllocationFailed {
                requested: std::mem::size_of::<usize>(),
            })?;
        self.marks.push(self.stack.len());
        Ok(())
    }

    fn duplicate_top(&mut self) -> Result<(), RestrictedPickleError> {
        let (nodes, payload_bytes) = self
            .stack
            .last()
            .map(|value| value_metrics(value, 1, self.limits.maximum_depth))
            .ok_or(RestrictedPickleError::InvalidStack("DUP needs a value"))??;
        self.charge_allocation(nodes, payload_bytes)?;
        let value = self
            .stack
            .last()
            .cloned()
            .ok_or(RestrictedPickleError::InvalidStack("DUP needs a value"))?;
        self.push_precharged(value)
    }

    fn pop(&mut self) -> Result<PickleValue, RestrictedPickleError> {
        self.stack
            .pop()
            .ok_or(RestrictedPickleError::InvalidStack("stack underflow"))
    }

    fn pop_mark(&mut self) -> Result<usize, RestrictedPickleError> {
        self.marks
            .pop()
            .ok_or(RestrictedPickleError::InvalidStack("MARK is missing"))
    }

    fn take_mark_values(
        &mut self,
        allocation_copies: u64,
    ) -> Result<Vec<PickleValue>, RestrictedPickleError> {
        let mark = self.pop_mark()?;
        if mark > self.stack.len() {
            return Err(RestrictedPickleError::InvalidStack(
                "MARK points outside the stack",
            ));
        }
        let value_count = self.stack.len().saturating_sub(mark);
        let allocation_nodes = u64::try_from(value_count)
            .unwrap_or(u64::MAX)
            .checked_mul(allocation_copies)
            .ok_or(ParserLimitError::Exceeded {
                kind: "pickle decoded allocation bytes",
                actual: u64::MAX,
                maximum: self.maximum_allocation_bytes(),
            })?;
        self.preflight_allocation(allocation_nodes, 0)?;
        Ok(self.stack.split_off(mark))
    }

    fn tuple_fixed(&mut self, length: usize) -> Result<(), RestrictedPickleError> {
        if self.stack.len() < length {
            return Err(RestrictedPickleError::InvalidStack(
                "tuple opcode has too few values",
            ));
        }
        let start = self.stack.len() - length;
        self.preflight_allocation(u64::try_from(length).unwrap_or(u64::MAX), 0)?;
        let values = self.stack.split_off(start);
        self.push(PickleValue::Tuple(values))
    }

    fn list_mut(&mut self) -> Result<&mut Vec<PickleValue>, RestrictedPickleError> {
        match self.stack.last_mut() {
            Some(PickleValue::List(values)) => Ok(values),
            _ => Err(RestrictedPickleError::InvalidStack(
                "list operation needs a list",
            )),
        }
    }

    fn dictionary_mut(
        &mut self,
    ) -> Result<&mut Vec<(PickleValue, PickleValue)>, RestrictedPickleError> {
        match self.stack.last_mut() {
            Some(PickleValue::Dictionary(values)) => Ok(values),
            _ => Err(RestrictedPickleError::InvalidStack(
                "dictionary operation needs a dictionary",
            )),
        }
    }

    fn set_mut(&mut self) -> Result<&mut Vec<PickleValue>, RestrictedPickleError> {
        match self.stack.last_mut() {
            Some(PickleValue::Set(values)) => Ok(values),
            _ => Err(RestrictedPickleError::InvalidStack(
                "set operation needs a set",
            )),
        }
    }

    fn append_list(&mut self, value: PickleValue) -> Result<(), RestrictedPickleError> {
        value_metrics(&value, 2, self.limits.maximum_depth)?;
        self.preflight_allocation(1, 0)?;
        let values = self.list_mut()?;
        values
            .try_reserve(1)
            .map_err(|_| RestrictedPickleError::AllocationFailed {
                requested: std::mem::size_of::<PickleValue>(),
            })?;
        values.push(value);
        Ok(())
    }

    fn extend_list(&mut self, additions: Vec<PickleValue>) -> Result<(), RestrictedPickleError> {
        for value in &additions {
            value_metrics(value, 2, self.limits.maximum_depth)?;
        }
        self.preflight_allocation(u64::try_from(additions.len()).unwrap_or(u64::MAX), 0)?;
        let values = self.list_mut()?;
        values.try_reserve(additions.len()).map_err(|_| {
            RestrictedPickleError::AllocationFailed {
                requested: additions
                    .len()
                    .saturating_mul(std::mem::size_of::<PickleValue>()),
            }
        })?;
        values.extend(additions);
        Ok(())
    }

    fn insert_dictionary(
        &mut self,
        key: PickleValue,
        value: PickleValue,
    ) -> Result<(), RestrictedPickleError> {
        value_metrics(&key, 2, self.limits.maximum_depth)?;
        value_metrics(&value, 2, self.limits.maximum_depth)?;
        self.preflight_allocation(2, 0)?;
        let values = self.dictionary_mut()?;
        values
            .try_reserve(1)
            .map_err(|_| RestrictedPickleError::AllocationFailed {
                requested: std::mem::size_of::<(PickleValue, PickleValue)>(),
            })?;
        values.push((key, value));
        Ok(())
    }

    fn extend_dictionary(
        &mut self,
        additions: Vec<(PickleValue, PickleValue)>,
    ) -> Result<(), RestrictedPickleError> {
        for (key, value) in &additions {
            value_metrics(key, 2, self.limits.maximum_depth)?;
            value_metrics(value, 2, self.limits.maximum_depth)?;
        }
        let allocation_nodes = u64::try_from(additions.len())
            .unwrap_or(u64::MAX)
            .checked_mul(2)
            .ok_or(ParserLimitError::Exceeded {
                kind: "pickle decoded allocation bytes",
                actual: u64::MAX,
                maximum: self.maximum_allocation_bytes(),
            })?;
        self.preflight_allocation(allocation_nodes, 0)?;
        let values = self.dictionary_mut()?;
        values.try_reserve(additions.len()).map_err(|_| {
            RestrictedPickleError::AllocationFailed {
                requested: additions
                    .len()
                    .saturating_mul(std::mem::size_of::<(PickleValue, PickleValue)>()),
            }
        })?;
        values.extend(additions);
        Ok(())
    }

    fn extend_set(&mut self, additions: Vec<PickleValue>) -> Result<(), RestrictedPickleError> {
        for value in &additions {
            value_metrics(value, 2, self.limits.maximum_depth)?;
        }
        self.preflight_allocation(u64::try_from(additions.len()).unwrap_or(u64::MAX), 0)?;
        let values = self.set_mut()?;
        values.try_reserve(additions.len()).map_err(|_| {
            RestrictedPickleError::AllocationFailed {
                requested: additions
                    .len()
                    .saturating_mul(std::mem::size_of::<PickleValue>()),
            }
        })?;
        values.extend(additions);
        Ok(())
    }

    fn memoize(&mut self, index: u32) -> Result<(), RestrictedPickleError> {
        if !self.memo.contains_key(&index) {
            self.limits.check(
                "pickle memo entries",
                u64::try_from(self.memo.len().saturating_add(1)).unwrap_or(u64::MAX),
                self.limits.maximum_metadata_values,
            )?;
        }
        let (nodes, payload_bytes) = self
            .stack
            .last()
            .map(|value| value_metrics(value, 1, self.limits.maximum_depth))
            .ok_or(RestrictedPickleError::InvalidStack(
                "memo operation needs a value",
            ))??;
        self.charge_allocation(nodes, payload_bytes)?;
        let value = self
            .stack
            .last()
            .cloned()
            .ok_or(RestrictedPickleError::InvalidStack(
                "memo operation needs a value",
            ))?;
        self.memo.insert(index, value);
        self.next_memo = self.next_memo.max(index.saturating_add(1));
        Ok(())
    }

    fn push_memo(&mut self, index: u32) -> Result<(), RestrictedPickleError> {
        let (nodes, payload_bytes) = self
            .memo
            .get(&index)
            .map(|value| value_metrics(value, 1, self.limits.maximum_depth))
            .ok_or(RestrictedPickleError::MissingMemo(index))??;
        self.charge_allocation(nodes, payload_bytes)?;
        let value = self
            .memo
            .get(&index)
            .cloned()
            .ok_or(RestrictedPickleError::MissingMemo(index))?;
        self.push_precharged(value)
    }

    fn reduce(&mut self) -> Result<(), RestrictedPickleError> {
        let arguments = self.pop()?;
        let callable = self.pop()?;
        let target = into_reduction_target(callable)?;
        validate_target(&target, TargetOperation::Reduce)?;
        self.charge_allocation(1, 0)?;
        self.push_precharged(PickleValue::Reduced {
            target,
            arguments: Box::new(arguments),
            state: None,
        })
    }

    fn build(&mut self) -> Result<(), RestrictedPickleError> {
        let state = self.pop()?;
        let mut object = self.pop()?;
        validate_target(
            object
                .reduction_target()
                .ok_or(RestrictedPickleError::InvalidStack(
                    "BUILD object has no allowlisted target",
                ))?,
            TargetOperation::Build,
        )?;
        self.charge_allocation(1, 0)?;
        match &mut object {
            PickleValue::Reduced {
                state: object_state,
                ..
            } => *object_state = Some(Box::new(state)),
            _ => {
                return Err(RestrictedPickleError::InvalidStack(
                    "BUILD requires a reduced object",
                ));
            }
        }
        self.push_precharged(object)
    }

    fn parse_text_integer(&mut self) -> Result<PickleValue, RestrictedPickleError> {
        let line = self.line()?;
        if line == b"00" {
            return Ok(PickleValue::Boolean(false));
        }
        if line == b"01" {
            return Ok(PickleValue::Boolean(true));
        }
        let text = std::str::from_utf8(line).map_err(|_| RestrictedPickleError::InvalidUtf8)?;
        let value = text
            .parse::<i128>()
            .map_err(|error| RestrictedPickleError::InvalidScalar(error.to_string()))?;
        Ok(PickleValue::Integer(value))
    }

    fn parse_text_long(&mut self) -> Result<PickleValue, RestrictedPickleError> {
        let line = self.line()?;
        let line = line.strip_suffix(b"L").unwrap_or(line);
        let text = std::str::from_utf8(line).map_err(|_| RestrictedPickleError::InvalidUtf8)?;
        let value = text
            .parse::<i128>()
            .map_err(|error| RestrictedPickleError::InvalidScalar(error.to_string()))?;
        Ok(PickleValue::Integer(value))
    }

    fn parse_binary_long(&mut self, length: usize) -> Result<PickleValue, RestrictedPickleError> {
        if length > 16 {
            return Err(RestrictedPickleError::InvalidScalar(
                "integer exceeds 128 bits".to_owned(),
            ));
        }
        let bytes = self.take(length)?;
        if bytes.is_empty() {
            return Ok(PickleValue::Integer(0));
        }
        let negative = bytes.last().is_some_and(|value| value & 0x80 != 0);
        let mut expanded = if negative { [0xff; 16] } else { [0; 16] };
        let destination = expanded.get_mut(..bytes.len()).ok_or_else(|| {
            RestrictedPickleError::InvalidScalar("integer width is invalid".to_owned())
        })?;
        destination.copy_from_slice(bytes);
        Ok(PickleValue::Integer(i128::from_le_bytes(expanded)))
    }

    fn utf8_string(&mut self, length: usize) -> Result<String, RestrictedPickleError> {
        self.check_name_length(length)?;
        let bytes = self.take(length)?;
        self.charge_payload(length)?;
        let value = std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| RestrictedPickleError::InvalidUtf8)?;
        Ok(value)
    }

    fn owned_bytes(&mut self, length: usize) -> Result<Vec<u8>, RestrictedPickleError> {
        self.limits.check(
            "pickle byte string",
            u64::try_from(length).unwrap_or(u64::MAX),
            self.limits.manifest_bytes,
        )?;
        self.ensure_readable(length)?;
        self.charge_payload(length)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(length)
            .map_err(|_| RestrictedPickleError::AllocationFailed { requested: length })?;
        result.extend_from_slice(self.take(length)?);
        Ok(result)
    }

    fn charge_payload(&mut self, length: usize) -> Result<(), RestrictedPickleError> {
        self.charge_allocation(0, u64::try_from(length).unwrap_or(u64::MAX))
    }

    fn charge_allocation(
        &mut self,
        nodes: u64,
        payload_bytes: u64,
    ) -> Result<(), RestrictedPickleError> {
        let maximum_allocation_bytes = self.maximum_allocation_bytes();
        self.value_count =
            self.value_count
                .checked_add(nodes)
                .ok_or(ParserLimitError::Exceeded {
                    kind: "pickle values",
                    actual: u64::MAX,
                    maximum: self.limits.maximum_metadata_values,
                })?;
        self.limits.check(
            "pickle values",
            self.value_count,
            self.limits.maximum_metadata_values,
        )?;
        let node_bytes = nodes
            .checked_mul(u64::try_from(std::mem::size_of::<PickleValue>()).unwrap_or(u64::MAX))
            .ok_or(ParserLimitError::Exceeded {
                kind: "pickle decoded allocation bytes",
                actual: u64::MAX,
                maximum: maximum_allocation_bytes,
            })?;
        let allocation =
            node_bytes
                .checked_add(payload_bytes)
                .ok_or(ParserLimitError::Exceeded {
                    kind: "pickle decoded allocation bytes",
                    actual: u64::MAX,
                    maximum: maximum_allocation_bytes,
                })?;
        self.allocation_bytes =
            self.allocation_bytes
                .checked_add(allocation)
                .ok_or(ParserLimitError::Exceeded {
                    kind: "pickle decoded allocation bytes",
                    actual: u64::MAX,
                    maximum: maximum_allocation_bytes,
                })?;
        self.limits.check(
            "pickle decoded allocation bytes",
            self.allocation_bytes,
            maximum_allocation_bytes,
        )?;
        Ok(())
    }

    fn preflight_allocation(
        &self,
        nodes: u64,
        payload_bytes: u64,
    ) -> Result<(), RestrictedPickleError> {
        let maximum_allocation_bytes = self.maximum_allocation_bytes();
        let node_bytes = nodes
            .checked_mul(u64::try_from(std::mem::size_of::<PickleValue>()).unwrap_or(u64::MAX))
            .ok_or(ParserLimitError::Exceeded {
                kind: "pickle decoded allocation bytes",
                actual: u64::MAX,
                maximum: maximum_allocation_bytes,
            })?;
        let requested =
            node_bytes
                .checked_add(payload_bytes)
                .ok_or(ParserLimitError::Exceeded {
                    kind: "pickle decoded allocation bytes",
                    actual: u64::MAX,
                    maximum: maximum_allocation_bytes,
                })?;
        let actual =
            self.allocation_bytes
                .checked_add(requested)
                .ok_or(ParserLimitError::Exceeded {
                    kind: "pickle decoded allocation bytes",
                    actual: u64::MAX,
                    maximum: maximum_allocation_bytes,
                })?;
        self.limits.check(
            "pickle decoded allocation bytes",
            actual,
            maximum_allocation_bytes,
        )?;
        Ok(())
    }

    fn maximum_allocation_bytes(&self) -> u64 {
        self.limits
            .manifest_bytes
            .saturating_mul(RESTRICTED_PICKLE_DECODED_ALLOCATION_MULTIPLIER)
    }

    fn check_name_length(&self, length: usize) -> Result<(), RestrictedPickleError> {
        self.limits.check(
            "pickle name bytes",
            u64::try_from(length).unwrap_or(u64::MAX),
            self.limits.maximum_name_bytes,
        )?;
        Ok(())
    }

    fn finish_frame_if_needed(&mut self) -> Result<(), RestrictedPickleError> {
        match self.frame_end {
            Some(frame_end) if self.cursor == frame_end => {
                self.frame_end = None;
                Ok(())
            }
            Some(frame_end) if self.cursor > frame_end => Err(
                RestrictedPickleError::InvalidScalar("pickle crossed a FRAME boundary".to_owned()),
            ),
            _ => Ok(()),
        }
    }

    fn read_boundary(&self) -> usize {
        self.frame_end.unwrap_or(self.bytes.len())
    }

    fn byte(&mut self) -> Result<u8, RestrictedPickleError> {
        self.ensure_readable(1)?;
        let value =
            self.bytes
                .get(self.cursor)
                .copied()
                .ok_or(RestrictedPickleError::Truncated {
                    offset: self.cursor,
                })?;
        self.cursor += 1;
        Ok(value)
    }

    fn array<const LENGTH: usize>(&mut self) -> Result<[u8; LENGTH], RestrictedPickleError> {
        let bytes = self.take(LENGTH)?;
        bytes
            .try_into()
            .map_err(|_| RestrictedPickleError::Truncated {
                offset: self.cursor,
            })
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], RestrictedPickleError> {
        self.ensure_readable(length)?;
        let end = self
            .cursor
            .checked_add(length)
            .ok_or(RestrictedPickleError::Truncated {
                offset: self.cursor,
            })?;
        let bytes = self
            .bytes
            .get(self.cursor..end)
            .ok_or(RestrictedPickleError::Truncated {
                offset: self.cursor,
            })?;
        self.cursor = end;
        Ok(bytes)
    }

    fn ensure_readable(&self, length: usize) -> Result<(), RestrictedPickleError> {
        let end = self
            .cursor
            .checked_add(length)
            .ok_or(RestrictedPickleError::Truncated {
                offset: self.cursor,
            })?;
        if self.frame_end.is_some_and(|frame_end| end > frame_end) {
            return Err(RestrictedPickleError::InvalidScalar(
                "pickle opcode crosses a FRAME boundary".to_owned(),
            ));
        }
        if end > self.bytes.len() {
            return Err(RestrictedPickleError::Truncated {
                offset: self.cursor,
            });
        }
        Ok(())
    }

    fn line(&mut self) -> Result<&'a [u8], RestrictedPickleError> {
        let boundary = self.read_boundary();
        let remaining =
            self.bytes
                .get(self.cursor..boundary)
                .ok_or(RestrictedPickleError::Truncated {
                    offset: self.cursor,
                })?;
        let newline = match remaining.iter().position(|byte| *byte == b'\n') {
            Some(newline) => newline,
            None if self.frame_end.is_some() => {
                return Err(RestrictedPickleError::InvalidScalar(
                    "pickle line crosses a FRAME boundary".to_owned(),
                ));
            }
            None => {
                return Err(RestrictedPickleError::Truncated {
                    offset: self.cursor,
                });
            }
        };
        let result = remaining
            .get(..newline)
            .ok_or(RestrictedPickleError::Truncated {
                offset: self.cursor,
            })?;
        self.cursor =
            self.cursor
                .checked_add(newline + 1)
                .ok_or(RestrictedPickleError::Truncated {
                    offset: self.cursor,
                })?;
        Ok(result)
    }

    fn length_u32(&mut self) -> Result<usize, RestrictedPickleError> {
        let value = u32::from_le_bytes(self.array::<4>()?);
        usize::try_from(value)
            .map_err(|_| RestrictedPickleError::InvalidScalar("length overflow".to_owned()))
    }

    fn length_u64(&mut self) -> Result<usize, RestrictedPickleError> {
        let value = u64::from_le_bytes(self.array::<8>()?);
        usize::try_from(value)
            .map_err(|_| RestrictedPickleError::InvalidScalar("length overflow".to_owned()))
    }
}

#[derive(Clone, Copy)]
enum TargetOperation {
    Global,
    Reduce,
    Build,
}

fn validate_target(target: &str, operation: TargetOperation) -> Result<(), RestrictedPickleError> {
    let allowed = ALLOWED_PICKLE_TARGETS.iter().any(|entry| {
        entry.target == target
            && match operation {
                TargetOperation::Global => entry.global,
                TargetOperation::Reduce => entry.reduce,
                TargetOperation::Build => entry.build,
            }
    });
    if allowed {
        Ok(())
    } else {
        let operation = match operation {
            TargetOperation::Global => "GLOBAL",
            TargetOperation::Reduce => "REDUCE",
            TargetOperation::Build => "BUILD",
        };
        Err(RestrictedPickleError::ForbiddenTarget {
            target: target.to_owned(),
            operation,
        })
    }
}

fn pair_values(
    values: Vec<PickleValue>,
) -> Result<Vec<(PickleValue, PickleValue)>, RestrictedPickleError> {
    if !values.len().is_multiple_of(2) {
        return Err(RestrictedPickleError::InvalidStack(
            "dictionary items are not key/value pairs",
        ));
    }
    let mut iterator = values.into_iter();
    let mut result = Vec::new();
    result.try_reserve_exact(iterator.len() / 2).map_err(|_| {
        RestrictedPickleError::AllocationFailed {
            requested: iterator
                .len()
                .saturating_div(2)
                .saturating_mul(std::mem::size_of::<(PickleValue, PickleValue)>()),
        }
    })?;
    while let Some(key) = iterator.next() {
        let value = iterator.next().ok_or(RestrictedPickleError::InvalidStack(
            "dictionary value is missing",
        ))?;
        result.push((key, value));
    }
    Ok(result)
}

fn into_reduction_target(value: PickleValue) -> Result<String, RestrictedPickleError> {
    match value {
        PickleValue::Global(target) | PickleValue::Reduced { target, .. } => Ok(target),
        _ => Err(RestrictedPickleError::InvalidStack(
            "REDUCE callable is not a global",
        )),
    }
}

fn value_metrics(
    value: &PickleValue,
    depth: u32,
    maximum_depth: u32,
) -> Result<(u64, u64), RestrictedPickleError> {
    if depth > maximum_depth {
        return Err(ParserLimitError::Exceeded {
            kind: "pickle value depth",
            actual: u64::from(depth),
            maximum: u64::from(maximum_depth),
        }
        .into());
    }
    let mut nodes = 1_u64;
    let mut payload_bytes = match value {
        PickleValue::Bytes(value) => u64::try_from(value.len()).unwrap_or(u64::MAX),
        PickleValue::String(value) | PickleValue::Global(value) => {
            u64::try_from(value.len()).unwrap_or(u64::MAX)
        }
        _ => 0,
    };
    match value {
        PickleValue::List(values) | PickleValue::Tuple(values) | PickleValue::Set(values) => {
            for value in values {
                add_value_metrics(value, depth, maximum_depth, &mut nodes, &mut payload_bytes)?;
            }
        }
        PickleValue::Dictionary(entries) => {
            for (key, value) in entries {
                add_value_metrics(key, depth, maximum_depth, &mut nodes, &mut payload_bytes)?;
                add_value_metrics(value, depth, maximum_depth, &mut nodes, &mut payload_bytes)?;
            }
        }
        PickleValue::Persistent(value) => {
            add_value_metrics(value, depth, maximum_depth, &mut nodes, &mut payload_bytes)?
        }
        PickleValue::Reduced {
            target,
            arguments,
            state,
        } => {
            payload_bytes = payload_bytes
                .checked_add(u64::try_from(target.len()).unwrap_or(u64::MAX))
                .ok_or(ParserLimitError::Exceeded {
                    kind: "pickle decoded allocation bytes",
                    actual: u64::MAX,
                    maximum: u64::MAX - 1,
                })?;
            add_value_metrics(
                arguments,
                depth,
                maximum_depth,
                &mut nodes,
                &mut payload_bytes,
            )?;
            if let Some(state) = state {
                add_value_metrics(state, depth, maximum_depth, &mut nodes, &mut payload_bytes)?;
            }
        }
        PickleValue::None
        | PickleValue::Boolean(_)
        | PickleValue::Integer(_)
        | PickleValue::FloatBits(_)
        | PickleValue::Bytes(_)
        | PickleValue::String(_)
        | PickleValue::Global(_) => {}
    }
    Ok((nodes, payload_bytes))
}

fn add_value_metrics(
    child: &PickleValue,
    parent_depth: u32,
    maximum_depth: u32,
    nodes: &mut u64,
    payload_bytes: &mut u64,
) -> Result<(), RestrictedPickleError> {
    let child_depth = parent_depth
        .checked_add(1)
        .ok_or(ParserLimitError::Exceeded {
            kind: "pickle value depth",
            actual: u64::MAX,
            maximum: u64::from(maximum_depth),
        })?;
    let (child_nodes, child_payload) = value_metrics(child, child_depth, maximum_depth)?;
    *nodes = nodes
        .checked_add(child_nodes)
        .ok_or(ParserLimitError::Exceeded {
            kind: "pickle values",
            actual: u64::MAX,
            maximum: u64::MAX - 1,
        })?;
    *payload_bytes =
        payload_bytes
            .checked_add(child_payload)
            .ok_or(ParserLimitError::Exceeded {
                kind: "pickle decoded allocation bytes",
                actual: u64::MAX,
                maximum: u64::MAX - 1,
            })?;
    Ok(())
}

fn value_string(value: PickleValue) -> Result<String, RestrictedPickleError> {
    match value {
        PickleValue::String(value) => Ok(value),
        _ => Err(RestrictedPickleError::InvalidStack(
            "STACK_GLOBAL needs module and name strings",
        )),
    }
}

fn join_global(
    module: &[u8],
    name: &[u8],
    maximum_name_bytes: u64,
) -> Result<String, RestrictedPickleError> {
    let module = std::str::from_utf8(module).map_err(|_| RestrictedPickleError::InvalidUtf8)?;
    let name = std::str::from_utf8(name).map_err(|_| RestrictedPickleError::InvalidUtf8)?;
    let target_length = module
        .len()
        .checked_add(name.len())
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| RestrictedPickleError::InvalidScalar("target length overflow".to_owned()))?;
    if u64::try_from(target_length).unwrap_or(u64::MAX) > maximum_name_bytes {
        return Err(ParserLimitError::Exceeded {
            kind: "pickle target bytes",
            actual: u64::try_from(target_length).unwrap_or(u64::MAX),
            maximum: maximum_name_bytes,
        }
        .into());
    }
    Ok(format!("{module}.{name}"))
}

fn parse_u32_line(line: &[u8]) -> Result<u32, RestrictedPickleError> {
    let text = std::str::from_utf8(line).map_err(|_| RestrictedPickleError::InvalidUtf8)?;
    text.parse::<u32>()
        .map_err(|error| RestrictedPickleError::InvalidScalar(error.to_string()))
}

fn parse_quoted_string(
    bytes: &[u8],
    maximum_name_bytes: u64,
) -> Result<String, RestrictedPickleError> {
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > maximum_name_bytes {
        return Err(ParserLimitError::Exceeded {
            kind: "pickle string bytes",
            actual: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            maximum: maximum_name_bytes,
        }
        .into());
    }
    let quote = bytes
        .first()
        .copied()
        .ok_or_else(|| RestrictedPickleError::InvalidScalar("quoted string is empty".to_owned()))?;
    if !matches!(quote, b'\'' | b'"') || bytes.last().copied() != Some(quote) {
        return Err(RestrictedPickleError::InvalidScalar(
            "STRING requires matching quotes".to_owned(),
        ));
    }
    let body = bytes.get(1..bytes.len().saturating_sub(1)).ok_or_else(|| {
        RestrictedPickleError::InvalidScalar("quoted string is malformed".to_owned())
    })?;
    decode_escaped_bytes(body)
}

fn decode_raw_unicode_escape(bytes: &[u8]) -> Result<String, RestrictedPickleError> {
    decode_escaped_bytes(bytes)
}

fn decode_escaped_bytes(bytes: &[u8]) -> Result<String, RestrictedPickleError> {
    let mut decoded = Vec::new();
    decoded
        .try_reserve(bytes.len())
        .map_err(|_| RestrictedPickleError::AllocationFailed {
            requested: bytes.len(),
        })?;
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte != b'\\' {
            decoded.push(byte);
            index += 1;
            continue;
        }
        let escaped = bytes.get(index + 1).copied().ok_or_else(|| {
            RestrictedPickleError::InvalidScalar("trailing string escape".to_owned())
        })?;
        match escaped {
            b'n' => decoded.push(b'\n'),
            b'r' => decoded.push(b'\r'),
            b't' => decoded.push(b'\t'),
            b'\\' => decoded.push(b'\\'),
            b'\'' => decoded.push(b'\''),
            b'"' => decoded.push(b'"'),
            b'x' => {
                let high = bytes.get(index + 2).copied().ok_or_else(|| {
                    RestrictedPickleError::InvalidScalar("short hexadecimal escape".to_owned())
                })?;
                let low = bytes.get(index + 3).copied().ok_or_else(|| {
                    RestrictedPickleError::InvalidScalar("short hexadecimal escape".to_owned())
                })?;
                decoded.push((hex(high)? << 4) | hex(low)?);
                index += 2;
            }
            _ => {
                return Err(RestrictedPickleError::InvalidScalar(format!(
                    "unsupported string escape \\{}",
                    char::from(escaped)
                )));
            }
        }
        index += 2;
    }
    String::from_utf8(decoded).map_err(|_| RestrictedPickleError::InvalidUtf8)
}

fn hex(value: u8) -> Result<u8, RestrictedPickleError> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err(RestrictedPickleError::InvalidScalar(
            "invalid hexadecimal escape".to_owned(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_and_structural_opcodes_parse() {
        let value = parse_restricted_pickle(
            b"\x80\x04}(\x8c\x01aK\x01\x8c\x01b](\x88Neu.",
            &ParserLimits::default(),
        );
        assert!(matches!(value, Ok(PickleValue::Dictionary(_))), "{value:?}");
    }

    #[test]
    fn tensor_rebuild_is_described_but_never_executed() {
        let bytes = b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n(K\x01tR.";
        let value = parse_restricted_pickle(bytes, &ParserLimits::default());
        assert!(matches!(
            value,
            Ok(PickleValue::Reduced { target, .. })
                if target == "torch._utils._rebuild_tensor_v2"
        ));
    }

    #[test]
    fn code_loading_and_unknown_targets_fail_closed() {
        let forbidden = parse_restricted_pickle(
            b"cos\nsystem\n(S'echo unsafe'\ntR.",
            &ParserLimits::default(),
        );
        assert!(matches!(
            forbidden,
            Err(RestrictedPickleError::ForbiddenTarget {
                operation: "GLOBAL",
                ..
            })
        ));
        let extension = parse_restricted_pickle(b"\x82\x01.", &ParserLimits::default());
        assert!(matches!(
            extension,
            Err(RestrictedPickleError::ForbiddenOpcode { opcode: 0x82, .. })
        ));
    }

    #[test]
    fn nesting_and_memo_clone_amplification_are_bounded() {
        let depth_limits = ParserLimits {
            maximum_depth: 2,
            ..ParserLimits::default()
        };
        assert!(matches!(
            parse_restricted_pickle(b"(((Nttt.", &depth_limits),
            Err(RestrictedPickleError::Limit(ParserLimitError::Exceeded {
                kind: "pickle mark depth",
                ..
            }))
        ));

        let value_limits = ParserLimits {
            maximum_metadata_values: 8,
            ..ParserLimits::default()
        };
        let mut amplified = vec![b']', b'q', 0];
        for _ in 0..8 {
            amplified.extend_from_slice(&[b'h', 0]);
        }
        amplified.push(b'.');
        assert!(matches!(
            parse_restricted_pickle(&amplified, &value_limits),
            Err(RestrictedPickleError::Limit(ParserLimitError::Exceeded {
                kind: "pickle values",
                ..
            }))
        ));
    }

    #[test]
    fn frame_boundaries_are_exact() {
        let mut valid = vec![0x80, 0x04, 0x95];
        valid.extend_from_slice(&2_u64.to_le_bytes());
        valid.extend_from_slice(b"N.");
        assert_eq!(
            parse_restricted_pickle(&valid, &ParserLimits::default()),
            Ok(PickleValue::None)
        );

        let mut crossing = vec![0x80, 0x04, 0x95];
        crossing.extend_from_slice(&1_u64.to_le_bytes());
        crossing.push(b'X');
        crossing.extend_from_slice(&0_u32.to_le_bytes());
        crossing.push(b'.');
        assert!(matches!(
            parse_restricted_pickle(&crossing, &ParserLimits::default()),
            Err(RestrictedPickleError::InvalidScalar(reason))
                if reason.contains("FRAME boundary")
        ));

        let mut one_byte_argument_crossing = vec![0x80, 0x04, 0x95];
        one_byte_argument_crossing.extend_from_slice(&1_u64.to_le_bytes());
        one_byte_argument_crossing.extend_from_slice(&[b'K', 1, b'.']);
        assert!(matches!(
            parse_restricted_pickle(&one_byte_argument_crossing, &ParserLimits::default()),
            Err(RestrictedPickleError::InvalidScalar(reason))
                if reason.contains("FRAME boundary")
        ));

        let mut line_crossing = vec![0x80, 0x04, 0x95];
        line_crossing.extend_from_slice(&2_u64.to_le_bytes());
        line_crossing.extend_from_slice(b"Px\n.");
        assert!(matches!(
            parse_restricted_pickle(&line_crossing, &ParserLimits::default()),
            Err(RestrictedPickleError::InvalidScalar(reason))
                if reason.contains("line crosses a FRAME boundary")
        ));

        let mut nested = vec![0x80, 0x04, 0x95];
        nested.extend_from_slice(&11_u64.to_le_bytes());
        nested.push(0x95);
        nested.extend_from_slice(&2_u64.to_le_bytes());
        nested.extend_from_slice(b"N.");
        assert!(matches!(
            parse_restricted_pickle(&nested, &ParserLimits::default()),
            Err(RestrictedPickleError::InvalidScalar(reason))
                if reason.contains("nested pickle FRAME")
        ));

        let mut zero = vec![0x80, 0x04, 0x95];
        zero.extend_from_slice(&0_u64.to_le_bytes());
        zero.extend_from_slice(b"N.");
        assert!(matches!(
            parse_restricted_pickle(&zero, &ParserLimits::default()),
            Err(RestrictedPickleError::InvalidScalar(reason))
                if reason.contains("must be non-zero")
        ));
    }

    #[test]
    fn stop_rejects_trailing_bytes() {
        assert!(matches!(
            parse_restricted_pickle(b"N.extra", &ParserLimits::default()),
            Err(RestrictedPickleError::InvalidScalar(reason))
                if reason.contains("trailing bytes")
        ));

        let mut frame_with_early_stop = vec![0x80, 0x04, 0x95];
        frame_with_early_stop.extend_from_slice(&2_u64.to_le_bytes());
        frame_with_early_stop.extend_from_slice(b".N");
        assert!(matches!(
            parse_restricted_pickle(&frame_with_early_stop, &ParserLimits::default()),
            Err(RestrictedPickleError::InvalidScalar(reason))
                if reason.contains("precedes the FRAME boundary")
        ));
    }

    #[test]
    fn textual_persistent_ids_are_name_bounded() {
        let limits = ParserLimits {
            maximum_name_bytes: 4,
            ..ParserLimits::default()
        };
        assert!(matches!(
            parse_restricted_pickle(b"P12345\n.", &limits),
            Err(RestrictedPickleError::Limit(ParserLimitError::Exceeded {
                kind: "pickle name bytes",
                ..
            }))
        ));
        assert!(matches!(
            parse_restricted_pickle(b"P1234\n.", &limits),
            Ok(PickleValue::Persistent(value))
                if matches!(value.as_ref(), PickleValue::String(value) if value == "1234")
        ));
    }

    #[test]
    fn container_and_memo_allocations_are_preflighted() {
        let limits = ParserLimits::default();
        let mut tuple_parser = PickleParser::new(b"", &limits, None);
        tuple_parser.stack.push(PickleValue::None);
        tuple_parser.allocation_bytes = tuple_parser.maximum_allocation_bytes();
        assert!(matches!(
            tuple_parser.tuple_fixed(1),
            Err(RestrictedPickleError::Limit(ParserLimitError::Exceeded {
                kind: "pickle decoded allocation bytes",
                ..
            }))
        ));
        assert_eq!(tuple_parser.stack, vec![PickleValue::None]);

        let mut memo_parser = PickleParser::new(b"", &limits, None);
        memo_parser.stack.push(PickleValue::List(Vec::new()));
        memo_parser.allocation_bytes = memo_parser.maximum_allocation_bytes();
        assert!(matches!(
            memo_parser.memoize(0),
            Err(RestrictedPickleError::Limit(ParserLimitError::Exceeded {
                kind: "pickle decoded allocation bytes",
                ..
            }))
        ));
        assert!(memo_parser.memo.is_empty());

        let mut list_parser = PickleParser::new(b"", &limits, None);
        list_parser.stack.push(PickleValue::List(Vec::new()));
        list_parser.allocation_bytes = list_parser.maximum_allocation_bytes();
        assert!(matches!(
            list_parser.append_list(PickleValue::None),
            Err(RestrictedPickleError::Limit(ParserLimitError::Exceeded {
                kind: "pickle decoded allocation bytes",
                ..
            }))
        ));
        assert_eq!(list_parser.stack, vec![PickleValue::List(Vec::new())]);
    }
}
