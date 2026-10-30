use crate::{DeviceId, Scalar, TensorError};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    F64,
    F32,
    F16,
    Bf16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
    Complex64,
    Complex128,
    Float8E4m3Fn,
    Float8E5m2,
    #[serde(rename = "float8_e4m3fnuz", alias = "float8_e4m3_fnuz")]
    Float8E4m3Fnuz,
    #[serde(rename = "float8_e5m2fnuz", alias = "float8_e5m2_fnuz")]
    Float8E5m2Fnuz,
    #[serde(rename = "float8_e8m0fnu", alias = "float8_e8m0_fnu")]
    Float8E8m0Fnu,
}

pub const ALL_DTYPES: [DType; 20] = [
    DType::F64,
    DType::F32,
    DType::F16,
    DType::Bf16,
    DType::I64,
    DType::I32,
    DType::I16,
    DType::I8,
    DType::U64,
    DType::U32,
    DType::U16,
    DType::U8,
    DType::Bool,
    DType::Complex64,
    DType::Complex128,
    DType::Float8E4m3Fn,
    DType::Float8E5m2,
    DType::Float8E4m3Fnuz,
    DType::Float8E5m2Fnuz,
    DType::Float8E8m0Fnu,
];

pub const CATALOG_MODEL_DTYPES: [(DType, &str); 9] = [
    (DType::Bf16, "COMFY-MODEL-0005"),
    (DType::F16, "COMFY-MODEL-0006"),
    (DType::F32, "COMFY-MODEL-0007"),
    (DType::F64, "COMFY-MODEL-0008"),
    (DType::Float8E4m3Fn, "COMFY-MODEL-0009"),
    (DType::Float8E4m3Fnuz, "COMFY-MODEL-0010"),
    (DType::Float8E5m2, "COMFY-MODEL-0011"),
    (DType::Float8E5m2Fnuz, "COMFY-MODEL-0012"),
    (DType::Float8E8m0Fnu, "COMFY-MODEL-0013"),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumericClass {
    Boolean,
    UnsignedInteger,
    SignedInteger,
    FloatingPoint,
    Complex,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecodedScalar {
    Boolean(bool),
    Signed(i64),
    Unsigned(u64),
    Real(f64),
    Complex { real: f64, imaginary: f64 },
}

impl DecodedScalar {
    pub const fn is_nonzero(self) -> bool {
        match self {
            Self::Boolean(value) => value,
            Self::Signed(value) => value != 0,
            Self::Unsigned(value) => value != 0,
            Self::Real(value) => value != 0.0,
            Self::Complex { real, imaginary } => real != 0.0 || imaginary != 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloatingPointInfo {
    bits: u16,
    epsilon: f64,
    maximum: f64,
    minimum: f64,
    smallest_normal: f64,
    resolution: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntegerInfo {
    bits: u16,
    minimum: i128,
    maximum: u128,
}

impl IntegerInfo {
    pub const fn bits(self) -> u16 {
        self.bits
    }

    pub const fn minimum(self) -> i128 {
        self.minimum
    }

    pub const fn maximum(self) -> u128 {
        self.maximum
    }
}

impl FloatingPointInfo {
    pub const fn bits(self) -> u16 {
        self.bits
    }

    pub const fn epsilon(self) -> f64 {
        self.epsilon
    }

    pub const fn maximum(self) -> f64 {
        self.maximum
    }

    pub const fn minimum(self) -> f64 {
        self.minimum
    }

    pub const fn smallest_normal(self) -> f64 {
        self.smallest_normal
    }

    pub const fn resolution(self) -> f64 {
        self.resolution
    }
}

impl DType {
    pub fn encode_decoded_scalar(
        self,
        value: DecodedScalar,
        operation: &str,
        device: DeviceId,
    ) -> Result<Vec<u8>, TensorError> {
        match (self, value) {
            (Self::Complex64, DecodedScalar::Complex { real, imaginary }) => {
                let mut bytes = Vec::with_capacity(8);
                bytes.extend_from_slice(&(real as f32).to_ne_bytes());
                bytes.extend_from_slice(&(imaginary as f32).to_ne_bytes());
                Ok(bytes)
            }
            (Self::Complex128, DecodedScalar::Complex { real, imaginary }) => {
                let mut bytes = Vec::with_capacity(16);
                bytes.extend_from_slice(&real.to_ne_bytes());
                bytes.extend_from_slice(&imaginary.to_ne_bytes());
                Ok(bytes)
            }
            (_, DecodedScalar::Complex { real, .. }) => {
                self.encode_scalar(Scalar::Float(real), operation, device)
            }
            (_, DecodedScalar::Boolean(value)) => {
                self.encode_scalar(Scalar::Boolean(value), operation, device)
            }
            (_, DecodedScalar::Signed(value)) => {
                self.encode_scalar(Scalar::Signed(value), operation, device)
            }
            (_, DecodedScalar::Unsigned(value)) => {
                self.encode_scalar(Scalar::Unsigned(value), operation, device)
            }
            (_, DecodedScalar::Real(value)) => {
                self.encode_scalar(Scalar::Float(value), operation, device)
            }
        }
    }

    pub fn integer_info(self) -> Result<IntegerInfo, TensorError> {
        let (minimum, maximum) = match self {
            Self::I64 => (i128::from(i64::MIN), i64::MAX as u128),
            Self::I32 => (i128::from(i32::MIN), i32::MAX as u128),
            Self::I16 => (i128::from(i16::MIN), i16::MAX as u128),
            Self::I8 => (i128::from(i8::MIN), i8::MAX as u128),
            Self::U64 => (0, u128::from(u64::MAX)),
            Self::U32 => (0, u128::from(u32::MAX)),
            Self::U16 => (0, u128::from(u16::MAX)),
            Self::U8 => (0, u128::from(u8::MAX)),
            _ => {
                return Err(TensorError::InvalidNumeric {
                    reason: format!("{} is not an integer dtype", self.catalog_name()),
                });
            }
        };
        Ok(IntegerInfo {
            bits: self.bit_width(),
            minimum,
            maximum,
        })
    }

    pub fn floating_point_info(self) -> Result<FloatingPointInfo, TensorError> {
        let info = match self {
            Self::F64 => FloatingPointInfo {
                bits: 64,
                epsilon: f64::EPSILON,
                maximum: f64::MAX,
                minimum: f64::MIN,
                smallest_normal: f64::MIN_POSITIVE,
                resolution: 1.0e-15,
            },
            Self::F32 => FloatingPointInfo {
                bits: 32,
                epsilon: f64::from(f32::EPSILON),
                maximum: f64::from(f32::MAX),
                minimum: f64::from(f32::MIN),
                smallest_normal: f64::from(f32::MIN_POSITIVE),
                resolution: 1.0e-6,
            },
            Self::F16 => FloatingPointInfo {
                bits: 16,
                epsilon: 0.000_976_562_5,
                maximum: 65_504.0,
                minimum: -65_504.0,
                smallest_normal: 0.000_061_035_156_25,
                resolution: 0.001,
            },
            Self::Bf16 => FloatingPointInfo {
                bits: 16,
                epsilon: 0.007_812_5,
                maximum: 3.389_531_389_251_535_5e38,
                minimum: -3.389_531_389_251_535_5e38,
                smallest_normal: 1.175_494_350_822_287_5e-38,
                resolution: 0.01,
            },
            Self::Float8E4m3Fn => FloatingPointInfo {
                bits: 8,
                epsilon: 0.125,
                maximum: 448.0,
                minimum: -448.0,
                smallest_normal: 0.015_625,
                resolution: 0.1,
            },
            Self::Float8E5m2 => FloatingPointInfo {
                bits: 8,
                epsilon: 0.25,
                maximum: 57_344.0,
                minimum: -57_344.0,
                smallest_normal: 0.000_061_035_156_25,
                resolution: 1.0,
            },
            Self::Float8E4m3Fnuz => FloatingPointInfo {
                bits: 8,
                epsilon: 0.125,
                maximum: 240.0,
                minimum: -240.0,
                smallest_normal: 0.007_812_5,
                resolution: 0.1,
            },
            Self::Float8E5m2Fnuz => FloatingPointInfo {
                bits: 8,
                epsilon: 0.25,
                maximum: 57_344.0,
                minimum: -57_344.0,
                smallest_normal: 0.000_030_517_578_125,
                resolution: 1.0,
            },
            Self::Float8E8m0Fnu => FloatingPointInfo {
                bits: 8,
                epsilon: 1.0,
                maximum: 1.701_411_834_604_692_3e38,
                minimum: 5.877_471_754_111_438e-39,
                smallest_normal: 5.877_471_754_111_438e-39,
                resolution: 1.0,
            },
            _ => {
                return Err(TensorError::InvalidNumeric {
                    reason: format!("{} is not a floating-point dtype", self.catalog_name()),
                });
            }
        };
        Ok(info)
    }

    pub const fn byte_width(self) -> u64 {
        match self {
            Self::F64 | Self::I64 | Self::U64 | Self::Complex64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::Bf16 | Self::I16 | Self::U16 => 2,
            Self::I8
            | Self::U8
            | Self::Bool
            | Self::Float8E4m3Fn
            | Self::Float8E5m2
            | Self::Float8E4m3Fnuz
            | Self::Float8E5m2Fnuz
            | Self::Float8E8m0Fnu => 1,
            Self::Complex128 => 16,
        }
    }

    pub const fn class(self) -> NumericClass {
        match self {
            Self::Bool => NumericClass::Boolean,
            Self::U64 | Self::U32 | Self::U16 | Self::U8 => NumericClass::UnsignedInteger,
            Self::I64 | Self::I32 | Self::I16 | Self::I8 => NumericClass::SignedInteger,
            Self::Complex64 | Self::Complex128 => NumericClass::Complex,
            Self::F64
            | Self::F32
            | Self::F16
            | Self::Bf16
            | Self::Float8E4m3Fn
            | Self::Float8E5m2
            | Self::Float8E4m3Fnuz
            | Self::Float8E5m2Fnuz
            | Self::Float8E8m0Fnu => NumericClass::FloatingPoint,
        }
    }

    pub const fn bit_width(self) -> u16 {
        (self.byte_width() * 8) as u16
    }

    pub const fn is_float8(self) -> bool {
        matches!(
            self,
            Self::Float8E4m3Fn
                | Self::Float8E5m2
                | Self::Float8E4m3Fnuz
                | Self::Float8E5m2Fnuz
                | Self::Float8E8m0Fnu
        )
    }

    pub const fn catalog_name(self) -> &'static str {
        match self {
            Self::F64 => "float64",
            Self::F32 => "float32",
            Self::F16 => "float16",
            Self::Bf16 => "bfloat16",
            Self::I64 => "int64",
            Self::I32 => "int32",
            Self::I16 => "int16",
            Self::I8 => "int8",
            Self::U64 => "uint64",
            Self::U32 => "uint32",
            Self::U16 => "uint16",
            Self::U8 => "uint8",
            Self::Bool => "bool",
            Self::Complex64 => "complex64",
            Self::Complex128 => "complex128",
            Self::Float8E4m3Fn => "float8_e4m3fn",
            Self::Float8E5m2 => "float8_e5m2",
            Self::Float8E4m3Fnuz => "float8_e4m3fnuz",
            Self::Float8E5m2Fnuz => "float8_e5m2fnuz",
            Self::Float8E8m0Fnu => "float8_e8m0fnu",
        }
    }

    pub fn encode_scalar(
        self,
        value: Scalar,
        operation: &str,
        device: DeviceId,
    ) -> Result<Vec<u8>, TensorError> {
        let real = scalar_to_f64(value);
        let bytes = match self {
            Self::F64 => real.to_ne_bytes().to_vec(),
            Self::F32 => (real as f32).to_ne_bytes().to_vec(),
            Self::F16 => f16::from_f64(real).to_bits().to_ne_bytes().to_vec(),
            Self::Bf16 => bf16::from_f64(real).to_bits().to_ne_bytes().to_vec(),
            Self::I64 => i64::try_from(scalar_to_i128(value)?)
                .map_err(|_| integer_conversion_error(self, value))?
                .to_ne_bytes()
                .to_vec(),
            Self::I32 => i32::try_from(scalar_to_i128(value)?)
                .map_err(|_| integer_conversion_error(self, value))?
                .to_ne_bytes()
                .to_vec(),
            Self::I16 => i16::try_from(scalar_to_i128(value)?)
                .map_err(|_| integer_conversion_error(self, value))?
                .to_ne_bytes()
                .to_vec(),
            Self::I8 => i8::try_from(scalar_to_i128(value)?)
                .map_err(|_| integer_conversion_error(self, value))?
                .to_ne_bytes()
                .to_vec(),
            Self::U64 => u64::try_from(scalar_to_u128(value)?)
                .map_err(|_| integer_conversion_error(self, value))?
                .to_ne_bytes()
                .to_vec(),
            Self::U32 => u32::try_from(scalar_to_u128(value)?)
                .map_err(|_| integer_conversion_error(self, value))?
                .to_ne_bytes()
                .to_vec(),
            Self::U16 => u16::try_from(scalar_to_u128(value)?)
                .map_err(|_| integer_conversion_error(self, value))?
                .to_ne_bytes()
                .to_vec(),
            Self::U8 => u8::try_from(scalar_to_u128(value)?)
                .map_err(|_| integer_conversion_error(self, value))?
                .to_ne_bytes()
                .to_vec(),
            Self::Bool => vec![u8::from(scalar_to_bool(value))],
            Self::Complex64 => {
                let mut bytes = (real as f32).to_ne_bytes().to_vec();
                bytes.extend_from_slice(&0_f32.to_ne_bytes());
                bytes
            }
            Self::Complex128 => {
                let mut bytes = real.to_ne_bytes().to_vec();
                bytes.extend_from_slice(&0_f64.to_ne_bytes());
                bytes
            }
            dtype if dtype.is_float8() => {
                let real = real as f32;
                vec![encode_float8(dtype, real).ok_or_else(|| {
                    TensorError::UnsupportedCapability {
                        operation: operation.to_owned(),
                        device,
                        reason: format!(
                            "{real:?} is not representable as checked {} storage",
                            dtype.catalog_name()
                        ),
                    }
                })?]
            }
            _ => {
                return Err(TensorError::UnsupportedCapability {
                    operation: operation.to_owned(),
                    device,
                    reason: format!("scalar encoding for {} is unavailable", self.catalog_name()),
                });
            }
        };
        Ok(bytes)
    }

    pub fn decode_scalar(self, bytes: &[u8]) -> Result<DecodedScalar, TensorError> {
        let expected =
            usize::try_from(self.byte_width()).map_err(|_| TensorError::ShapeOverflow)?;
        if bytes.len() != expected {
            return Err(TensorError::StorageLength {
                expected: self.byte_width(),
                actual: u64::try_from(bytes.len()).map_err(|_| TensorError::ShapeOverflow)?,
            });
        }
        match self {
            Self::F64 => Ok(DecodedScalar::Real(f64::from_ne_bytes(array(bytes)?))),
            Self::F32 => Ok(DecodedScalar::Real(f64::from(f32::from_ne_bytes(array(
                bytes,
            )?)))),
            Self::F16 => Ok(DecodedScalar::Real(f64::from(f16::from_bits(
                u16::from_ne_bytes(array(bytes)?),
            )))),
            Self::Bf16 => Ok(DecodedScalar::Real(f64::from(bf16::from_bits(
                u16::from_ne_bytes(array(bytes)?),
            )))),
            Self::I64 => Ok(DecodedScalar::Signed(i64::from_ne_bytes(array(bytes)?))),
            Self::I32 => Ok(DecodedScalar::Signed(i64::from(i32::from_ne_bytes(array(
                bytes,
            )?)))),
            Self::I16 => Ok(DecodedScalar::Signed(i64::from(i16::from_ne_bytes(array(
                bytes,
            )?)))),
            Self::I8 => Ok(DecodedScalar::Signed(i64::from(i8::from_ne_bytes(array(
                bytes,
            )?)))),
            Self::U64 => Ok(DecodedScalar::Unsigned(u64::from_ne_bytes(array(bytes)?))),
            Self::U32 => Ok(DecodedScalar::Unsigned(u64::from(u32::from_ne_bytes(
                array(bytes)?,
            )))),
            Self::U16 => Ok(DecodedScalar::Unsigned(u64::from(u16::from_ne_bytes(
                array(bytes)?,
            )))),
            Self::U8 => Ok(DecodedScalar::Unsigned(u64::from(bytes[0]))),
            Self::Bool => Ok(DecodedScalar::Boolean(bytes[0] != 0)),
            Self::Complex64 => {
                let real = f32::from_ne_bytes(array(&bytes[..4])?);
                let imaginary = f32::from_ne_bytes(array(&bytes[4..])?);
                Ok(DecodedScalar::Complex {
                    real: f64::from(real),
                    imaginary: f64::from(imaginary),
                })
            }
            Self::Complex128 => {
                let real = f64::from_ne_bytes(array(&bytes[..8])?);
                let imaginary = f64::from_ne_bytes(array(&bytes[8..])?);
                Ok(DecodedScalar::Complex { real, imaginary })
            }
            dtype if dtype.is_float8() => Ok(DecodedScalar::Real(f64::from(decode_float8(
                dtype, bytes[0],
            )))),
            _ => Err(TensorError::InvalidNumeric {
                reason: format!("no decoder exists for {}", self.catalog_name()),
            }),
        }
    }
}

pub fn decode_float8(dtype: DType, bits: u8) -> f32 {
    match dtype {
        DType::Float8E4m3Fn => decode_signed_float8(bits, 4, 3, 7, Float8Special::FiniteNan),
        DType::Float8E4m3Fnuz => {
            decode_signed_float8(bits, 4, 3, 8, Float8Special::UnsignedZeroNan)
        }
        DType::Float8E5m2 => decode_signed_float8(bits, 5, 2, 15, Float8Special::Ieee),
        DType::Float8E5m2Fnuz => {
            decode_signed_float8(bits, 5, 2, 16, Float8Special::UnsignedZeroNan)
        }
        DType::Float8E8m0Fnu => {
            if bits == u8::MAX {
                f32::NAN
            } else {
                2.0_f32.powi(i32::from(bits) - 127)
            }
        }
        _ => f32::NAN,
    }
}

pub fn encode_float8(dtype: DType, value: f32) -> Option<u8> {
    if !dtype.is_float8() {
        return None;
    }
    if value.is_nan() {
        return Some(match dtype {
            DType::Float8E4m3Fn => 0x7f,
            DType::Float8E5m2 => 0x7f,
            DType::Float8E4m3Fnuz | DType::Float8E5m2Fnuz => 0x80,
            DType::Float8E8m0Fnu => 0xff,
            _ => return None,
        });
    }
    if value.is_infinite() {
        return match dtype {
            DType::Float8E5m2 => Some(if value.is_sign_negative() { 0xfc } else { 0x7c }),
            _ => None,
        };
    }
    if value == 0.0 {
        return match dtype {
            DType::Float8E4m3Fn | DType::Float8E5m2 => {
                Some(if value.is_sign_negative() { 0x80 } else { 0x00 })
            }
            DType::Float8E4m3Fnuz | DType::Float8E5m2Fnuz => Some(0x00),
            DType::Float8E8m0Fnu => None,
            _ => None,
        };
    }
    let maximum = match dtype {
        DType::Float8E4m3Fn => 448.0,
        DType::Float8E4m3Fnuz => 240.0,
        DType::Float8E5m2 | DType::Float8E5m2Fnuz => 57_344.0,
        DType::Float8E8m0Fnu => 2.0_f32.powi(127),
        _ => return None,
    };
    if value.abs() > maximum {
        return None;
    }
    if dtype == DType::Float8E8m0Fnu && (value <= 0.0 || value < 2.0_f32.powi(-127)) {
        return None;
    }
    let mut best = None;
    let mut best_distance = f32::INFINITY;
    for candidate in u8::MIN..=u8::MAX {
        let decoded = decode_float8(dtype, candidate);
        if decoded.is_nan() || (!value.is_infinite() && decoded.is_infinite()) {
            continue;
        }
        let distance = (decoded - value).abs();
        if distance < best_distance
            || (distance == best_distance
                && best.is_none_or(|current: u8| candidate & 1 == 0 && current & 1 != 0))
        {
            best = Some(candidate);
            best_distance = distance;
        }
    }
    best
}

#[derive(Clone, Copy)]
enum Float8Special {
    Ieee,
    FiniteNan,
    UnsignedZeroNan,
}

fn decode_signed_float8(
    bits: u8,
    exponent_bits: u32,
    mantissa_bits: u32,
    bias: i32,
    special: Float8Special,
) -> f32 {
    if matches!(special, Float8Special::UnsignedZeroNan) && bits == 0x80 {
        return f32::NAN;
    }
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent_mask = (1_u8 << exponent_bits) - 1;
    let mantissa_mask = (1_u8 << mantissa_bits) - 1;
    let exponent = (bits >> mantissa_bits) & exponent_mask;
    let mantissa = bits & mantissa_mask;
    if matches!(special, Float8Special::Ieee) && exponent == exponent_mask {
        return if mantissa == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    if matches!(special, Float8Special::FiniteNan)
        && exponent == exponent_mask
        && mantissa == mantissa_mask
    {
        return f32::NAN;
    }
    if exponent == 0 {
        if mantissa == 0 {
            return sign * 0.0;
        }
        return sign
            * f32::from(mantissa)
            * 2.0_f32.powi(1 - bias - i32::try_from(mantissa_bits).unwrap_or(0));
    }
    sign * (1.0 + f32::from(mantissa) / f32::from(1_u8 << mantissa_bits))
        * 2.0_f32.powi(i32::from(exponent) - bias)
}

fn array<const N: usize>(bytes: &[u8]) -> Result<[u8; N], TensorError> {
    bytes.try_into().map_err(|_| TensorError::StorageLength {
        expected: N as u64,
        actual: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
    })
}

fn scalar_to_f64(value: Scalar) -> f64 {
    match value {
        Scalar::Boolean(value) => f64::from(u8::from(value)),
        Scalar::Signed(value) => value as f64,
        Scalar::Unsigned(value) => value as f64,
        Scalar::Float(value) => value,
    }
}

fn scalar_to_bool(value: Scalar) -> bool {
    match value {
        Scalar::Boolean(value) => value,
        Scalar::Signed(value) => value != 0,
        Scalar::Unsigned(value) => value != 0,
        Scalar::Float(value) => value != 0.0,
    }
}

fn scalar_to_i128(value: Scalar) -> Result<i128, TensorError> {
    match value {
        Scalar::Boolean(value) => Ok(i128::from(u8::from(value))),
        Scalar::Signed(value) => Ok(i128::from(value)),
        Scalar::Unsigned(value) => Ok(i128::from(value)),
        Scalar::Float(value) if value.is_finite() => Ok(value.trunc() as i128),
        Scalar::Float(_) => Err(TensorError::InvalidNumeric {
            reason: "a non-finite scalar cannot be converted to an integer".to_owned(),
        }),
    }
}

fn scalar_to_u128(value: Scalar) -> Result<u128, TensorError> {
    match value {
        Scalar::Boolean(value) => Ok(u128::from(u8::from(value))),
        Scalar::Signed(value) => u128::try_from(value).map_err(|_| TensorError::InvalidNumeric {
            reason: format!("negative scalar {value} cannot be converted to an unsigned dtype"),
        }),
        Scalar::Unsigned(value) => Ok(u128::from(value)),
        Scalar::Float(value) if value.is_finite() && value >= 0.0 => Ok(value.trunc() as u128),
        Scalar::Float(_) => Err(TensorError::InvalidNumeric {
            reason: "a negative or non-finite scalar cannot be converted to an unsigned dtype"
                .to_owned(),
        }),
    }
}

fn integer_conversion_error(dtype: DType, value: Scalar) -> TensorError {
    TensorError::InvalidNumeric {
        reason: format!("scalar {value:?} is outside the range of {dtype:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoded_scalar_owns_truth_semantics_for_every_numeric_class() {
        assert!(!DecodedScalar::Boolean(false).is_nonzero());
        assert!(DecodedScalar::Boolean(true).is_nonzero());
        assert!(!DecodedScalar::Signed(0).is_nonzero());
        assert!(DecodedScalar::Signed(-1).is_nonzero());
        assert!(!DecodedScalar::Unsigned(0).is_nonzero());
        assert!(DecodedScalar::Unsigned(1).is_nonzero());
        assert!(!DecodedScalar::Real(-0.0).is_nonzero());
        assert!(DecodedScalar::Real(f64::NAN).is_nonzero());
        assert!(
            !DecodedScalar::Complex {
                real: -0.0,
                imaginary: 0.0,
            }
            .is_nonzero()
        );
        assert!(
            DecodedScalar::Complex {
                real: 0.0,
                imaginary: f64::NAN,
            }
            .is_nonzero()
        );
    }

    #[test]
    fn catalog_float8_boundaries_decode_exactly() {
        assert_eq!(decode_float8(DType::Float8E4m3Fn, 0x7e), 448.0);
        assert!(decode_float8(DType::Float8E4m3Fn, 0x7f).is_nan());
        assert_eq!(decode_float8(DType::Float8E4m3Fnuz, 0x7f), 240.0);
        assert!(decode_float8(DType::Float8E4m3Fnuz, 0x80).is_nan());
        assert_eq!(decode_float8(DType::Float8E5m2, 0x7b), 57_344.0);
        assert!(decode_float8(DType::Float8E5m2, 0x7c).is_infinite());
        assert_eq!(decode_float8(DType::Float8E5m2Fnuz, 0x7f), 57_344.0);
        assert_eq!(decode_float8(DType::Float8E8m0Fnu, 127), 1.0);
        assert!(decode_float8(DType::Float8E8m0Fnu, 255).is_nan());
        assert_eq!(encode_float8(DType::Float8E4m3Fn, -0.0), Some(0x80));
        assert_eq!(encode_float8(DType::Float8E4m3Fnuz, -0.0), Some(0x00));
        assert_eq!(encode_float8(DType::Float8E4m3Fn, 449.0), None);
    }

    #[test]
    fn scalar_codecs_round_trip_and_reject_invalid_integer_casts() {
        let encoded = DType::F16
            .encode_scalar(Scalar::Float(1.5), "test", DeviceId::CPU)
            .expect("f16 encoding is supported");
        assert_eq!(
            DType::F16.decode_scalar(&encoded),
            Ok(DecodedScalar::Real(1.5))
        );
        assert!(matches!(
            DType::U8.encode_scalar(Scalar::Signed(-1), "test", DeviceId::CPU),
            Err(TensorError::InvalidNumeric { .. })
        ));
        let float8 = DType::Float8E4m3Fn
            .encode_scalar(Scalar::Float(1.5), "test", DeviceId::CPU)
            .expect("finite float8 value is representable");
        assert_eq!(
            DType::Float8E4m3Fn.decode_scalar(&float8),
            Ok(DecodedScalar::Real(1.5))
        );
    }

    #[test]
    fn floating_point_info_is_owned_by_the_canonical_dtype_catalog() {
        let f16 = DType::F16
            .floating_point_info()
            .expect("f16 metadata is cataloged");
        assert_eq!(f16.bits(), 16);
        assert_eq!(f16.epsilon(), 0.000_976_562_5);
        assert_eq!(f16.maximum(), 65_504.0);
        assert_eq!(f16.minimum(), -65_504.0);
        assert_eq!(f16.smallest_normal(), 0.000_061_035_156_25);
        assert_eq!(f16.resolution(), 0.001);

        let unsigned_float8 = DType::Float8E8m0Fnu
            .floating_point_info()
            .expect("e8m0 metadata is cataloged");
        assert_eq!(unsigned_float8.bits(), 8);
        assert!(unsigned_float8.minimum().is_sign_positive());
        assert_eq!(unsigned_float8.minimum(), unsigned_float8.smallest_normal());
        assert!(matches!(
            DType::I32.floating_point_info(),
            Err(TensorError::InvalidNumeric { .. })
        ));
    }
}
