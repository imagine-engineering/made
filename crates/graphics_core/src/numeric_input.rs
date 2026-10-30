use std::fmt;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NumberInputConstraints {
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub is_integer: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NumberInputError {
    Parse(String),
    Evaluation(String),
    NonReal,
    NotANumber,
}

impl fmt::Display for NumberInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => write!(formatter, "Invalid numeric expression: {error}"),
            Self::Evaluation(error) => write!(formatter, "Numeric evaluation failed: {error}"),
            Self::NonReal => formatter.write_str("Numeric expression must produce a real number"),
            Self::NotANumber => formatter.write_str("Numeric expression produced NaN"),
        }
    }
}

impl std::error::Error for NumberInputError {}

pub fn evaluate_number_input(
    expression: &str,
    constraints: NumberInputConstraints,
) -> Result<f64, NumberInputError> {
    let value = graphite_math_parser::evaluate(expression)
        .map_err(|error| NumberInputError::Parse(error.to_string()))?
        .map_err(|error| NumberInputError::Evaluation(error.to_string()))?;
    let real = value.as_real().ok_or(NumberInputError::NonReal)?;
    if real.is_nan() {
        return Err(NumberInputError::NotANumber);
    }
    let mut validated = real;
    if let Some(minimum) = constraints.minimum {
        validated = validated.max(minimum);
    }
    if let Some(maximum) = constraints.maximum {
        validated = validated.min(maximum);
    }
    if constraints.is_integer {
        validated = validated.round();
    }
    Ok(validated)
}
