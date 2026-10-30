use crate::{DType, NumericClass, TensorError};

pub fn promote_types(left: DType, right: DType) -> Result<DType, TensorError> {
    if left == right {
        return Ok(left);
    }
    if left.is_float8() || right.is_float8() {
        return Err(TensorError::UnsupportedCapability {
            operation: "dtype promotion".to_owned(),
            device: crate::DeviceId::CPU,
            reason: format!(
                "PyTorch does not define implicit mixed promotion for {} and {}",
                left.catalog_name(),
                right.catalog_name()
            ),
        });
    }
    if left == DType::Bool {
        return Ok(right);
    }
    if right == DType::Bool {
        return Ok(left);
    }
    if matches!(left.class(), NumericClass::Complex)
        || matches!(right.class(), NumericClass::Complex)
    {
        return Ok(
            if left == DType::Complex128
                || right == DType::Complex128
                || left == DType::F64
                || right == DType::F64
            {
                DType::Complex128
            } else {
                DType::Complex64
            },
        );
    }
    if matches!(left.class(), NumericClass::FloatingPoint)
        || matches!(right.class(), NumericClass::FloatingPoint)
    {
        return promote_floats(left, right);
    }
    promote_integers(left, right)
}

pub fn can_cast_without_loss(source: DType, destination: DType) -> bool {
    if source == destination || source == DType::Bool {
        return true;
    }
    if source.is_float8() {
        return matches!(
            destination,
            DType::F16
                | DType::Bf16
                | DType::F32
                | DType::F64
                | DType::Complex64
                | DType::Complex128
        );
    }
    match (source.class(), destination.class()) {
        (NumericClass::UnsignedInteger, NumericClass::UnsignedInteger)
        | (NumericClass::SignedInteger, NumericClass::SignedInteger) => {
            destination.bit_width() >= source.bit_width()
        }
        (NumericClass::UnsignedInteger, NumericClass::SignedInteger) => {
            destination.bit_width() > source.bit_width()
        }
        (NumericClass::SignedInteger, NumericClass::FloatingPoint)
        | (NumericClass::UnsignedInteger, NumericClass::FloatingPoint) => {
            exact_integer_bits(destination) >= source.bit_width()
        }
        (NumericClass::FloatingPoint, NumericClass::FloatingPoint) => {
            float_rank(destination) >= float_rank(source)
        }
        (_, NumericClass::Complex) => destination == DType::Complex128 || source != DType::F64,
        _ => false,
    }
}

fn promote_floats(left: DType, right: DType) -> Result<DType, TensorError> {
    if left == DType::F64 || right == DType::F64 {
        return Ok(DType::F64);
    }
    if left == DType::F32 || right == DType::F32 {
        return Ok(DType::F32);
    }
    if (left == DType::F16 && right == DType::Bf16) || (left == DType::Bf16 && right == DType::F16)
    {
        return Ok(DType::F32);
    }
    if left == DType::Bf16 || right == DType::Bf16 {
        return Ok(DType::Bf16);
    }
    if left == DType::F16 || right == DType::F16 {
        return Ok(DType::F16);
    }
    Err(TensorError::InvalidNumeric {
        reason: format!(
            "no floating promotion exists for {} and {}",
            left.catalog_name(),
            right.catalog_name()
        ),
    })
}

fn promote_integers(left: DType, right: DType) -> Result<DType, TensorError> {
    let left_signed = matches!(left.class(), NumericClass::SignedInteger);
    let right_signed = matches!(right.class(), NumericClass::SignedInteger);
    let left_bits = left.bit_width();
    let right_bits = right.bit_width();
    if left_signed == right_signed {
        return integer_dtype(left_signed, left_bits.max(right_bits));
    }
    let signed_bits = if left_signed { left_bits } else { right_bits };
    let unsigned_bits = if left_signed { right_bits } else { left_bits };
    let required = signed_bits.max(unsigned_bits.saturating_add(1));
    for bits in [8, 16, 32, 64] {
        if bits >= required {
            return integer_dtype(true, bits);
        }
    }
    Ok(DType::F64)
}

fn integer_dtype(signed: bool, bits: u16) -> Result<DType, TensorError> {
    match (signed, bits) {
        (true, 8) => Ok(DType::I8),
        (true, 16) => Ok(DType::I16),
        (true, 32) => Ok(DType::I32),
        (true, 64) => Ok(DType::I64),
        (false, 8) => Ok(DType::U8),
        (false, 16) => Ok(DType::U16),
        (false, 32) => Ok(DType::U32),
        (false, 64) => Ok(DType::U64),
        _ => Err(TensorError::InvalidNumeric {
            reason: format!("unsupported integer width {bits}"),
        }),
    }
}

const fn float_rank(dtype: DType) -> u8 {
    match dtype {
        DType::Float8E4m3Fn
        | DType::Float8E5m2
        | DType::Float8E4m3Fnuz
        | DType::Float8E5m2Fnuz
        | DType::Float8E8m0Fnu => 0,
        DType::F16 => 1,
        DType::Bf16 => 2,
        DType::F32 => 3,
        DType::F64 => 4,
        _ => 0,
    }
}

const fn exact_integer_bits(dtype: DType) -> u16 {
    match dtype {
        DType::F16 => 11,
        DType::Bf16 => 8,
        DType::F32 => 24,
        DType::F64 => 53,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pytorch_style_promotion_covers_boundaries() {
        assert_eq!(promote_types(DType::Bool, DType::I8), Ok(DType::I8));
        assert_eq!(promote_types(DType::U8, DType::I8), Ok(DType::I16));
        assert_eq!(promote_types(DType::U32, DType::I32), Ok(DType::I64));
        assert_eq!(promote_types(DType::U64, DType::I64), Ok(DType::F64));
        assert_eq!(promote_types(DType::F16, DType::Bf16), Ok(DType::F32));
        assert_eq!(
            promote_types(DType::F32, DType::Complex64),
            Ok(DType::Complex64)
        );
        assert!(matches!(
            promote_types(DType::Float8E4m3Fn, DType::F16),
            Err(TensorError::UnsupportedCapability { .. })
        ));
    }
}
