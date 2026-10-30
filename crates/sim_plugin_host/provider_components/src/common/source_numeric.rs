use num_rational::Ratio;
use num_traits::ToPrimitive;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PositiveIntegerRatioError {
    NonPositivePart,
    ConversionUnavailable,
}

pub(crate) fn positive_integer_ratio_to_f64(
    numerator: u64,
    denominator: u64,
) -> Result<f64, PositiveIntegerRatioError> {
    if numerator == 0 || denominator == 0 {
        return Err(PositiveIntegerRatioError::NonPositivePart);
    }
    // Rounding each integer before division loses source admissions near 0.25
    // and 4. The rational owner rounds the quotient once, without a GCD pass.
    Ratio::<u64>::new_raw(numerator, denominator)
        .to_f64()
        .filter(|value| value.is_finite())
        .ok_or(PositiveIntegerRatioError::ConversionUnavailable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_integer_division_preserves_source_rounding_boundaries()
    -> Result<(), PositiveIntegerRatioError> {
        for (numerator, denominator, expected_bits) in [
            (
                4_611_686_018_427_388_417,
                1_152_921_504_606_847_104,
                0x4010000000000000,
            ),
            (
                1_152_921_504_606_847_104,
                4_611_686_018_427_388_417,
                0x3fd0000000000000,
            ),
            ((1 << 53) + 1, 1 << 51, 0x4010000000000000),
            ((1 << 53) + 2, 1 << 51, 0x4010000000000001),
            ((1 << 54) - 1, 1 << 56, 0x3fd0000000000000),
            ((1 << 54) - 2, 1 << 56, 0x3fcfffffffffffff),
            (1, 3, 0x3fd5555555555555),
            (u64::MAX, u64::MAX, 0x3ff0000000000000),
            (1, 18446744073709551615, 0x3bf0000000000000),
            (18446744073709551615, 1, 0x43f0000000000000),
            (9007199254740992, 9007199254740992, 0x3ff0000000000000),
            (9223372036854775807, 9223372036854775807, 0x3ff0000000000000),
            (9223372036854775808, 9223372036854775808, 0x3ff0000000000000),
        ] {
            assert_eq!(
                positive_integer_ratio_to_f64(numerator, denominator)?.to_bits(),
                expected_bits
            );
        }
        Ok(())
    }

    #[test]
    fn positive_integer_division_rejects_zero_parts() {
        for (numerator, denominator) in [(0, 0), (1, 0), (0, 1)] {
            assert_eq!(
                positive_integer_ratio_to_f64(numerator, denominator),
                Err(PositiveIntegerRatioError::NonPositivePart)
            );
        }
    }
}
