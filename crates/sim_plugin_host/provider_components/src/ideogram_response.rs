use serde::Deserialize;
use std::fmt;

#[derive(Deserialize)]
pub(crate) struct GenerateResponse {
    pub(crate) data: Option<Vec<ImageDatum>>,
    #[serde(default, deserialize_with = "deserialize_response_created")]
    pub(crate) created: Option<speedate::DateTime>,
}
#[derive(Deserialize)]
pub(crate) struct ImageDatum {
    pub(crate) url: Option<String>,
    #[serde(default, deserialize_with = "deserialize_response_safety")]
    pub(crate) is_image_safe: Option<bool>,
    pub(crate) prompt: Option<String>,
    pub(crate) resolution: Option<String>,
    #[serde(default, deserialize_with = "deserialize_response_seed")]
    pub(crate) seed: Option<i64>,
    pub(crate) style_type: Option<String>,
}
fn deserialize_response_safety<'de, Deserializer>(
    deserializer: Deserializer,
) -> Result<Option<bool>, Deserializer::Error>
where
    Deserializer: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum ResponseSafety {
        Boolean(bool),
        Text(String),
        Number(serde_json::Number),
    }
    Option::<ResponseSafety>::deserialize(deserializer)?
        .map(|safety| match safety {
            ResponseSafety::Boolean(value) => Ok(value),
            ResponseSafety::Text(value) => {
                if ["0", "f", "n", "no", "off", "false"]
                    .iter()
                    .any(|token| value.eq_ignore_ascii_case(token))
                {
                    Ok(false)
                } else if ["1", "t", "y", "on", "yes", "true"]
                    .iter()
                    .any(|token| value.eq_ignore_ascii_case(token))
                {
                    Ok(true)
                } else {
                    Err(serde::de::Error::custom("is_image_safe must be a boolean"))
                }
            }
            ResponseSafety::Number(value) => {
                if value.as_f64() == Some(0.0) {
                    Ok(false)
                } else if value.as_f64() == Some(1.0) {
                    Ok(true)
                } else {
                    Err(serde::de::Error::custom("is_image_safe must be a boolean"))
                }
            }
        })
        .transpose()
}
fn deserialize_response_seed<'de, Deserializer>(
    deserializer: Deserializer,
) -> Result<Option<i64>, Deserializer::Error>
where
    Deserializer: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum ResponseSeed {
        Integer(i64),
        Decimal(String),
        Boolean(bool),
        Float(f64),
    }
    Option::<ResponseSeed>::deserialize(deserializer)?
        .map(|seed| match seed {
            ResponseSeed::Integer(value) => Ok(value),
            ResponseSeed::Decimal(value) => {
                parse_response_seed_text(&value).map_err(serde::de::Error::custom)
            }
            ResponseSeed::Boolean(value) => Ok(i64::from(value)),
            ResponseSeed::Float(value) => {
                if !value.is_finite() || value.fract() != 0.0 {
                    return Err(serde::de::Error::custom("seed must be a finite integer"));
                }
                if value <= i64::MIN as f64 {
                    return Err(serde::de::Error::custom("seed is outside the i64 range"));
                }
                num_traits::ToPrimitive::to_i64(&value)
                    .ok_or_else(|| serde::de::Error::custom("seed is outside the i64 range"))
            }
        })
        .transpose()
}
fn parse_response_seed_text(value: &str) -> Result<i64, std::num::ParseIntError> {
    let original_error = match value.parse::<i64>() {
        Ok(integer) => return Ok(integer),
        Err(error) => error,
    };
    let mut normalized = value.trim();
    if let Some(suffix) = normalized.strip_prefix('+') {
        if suffix.starts_with('-') {
            return Err(original_error);
        }
        normalized = suffix;
    }
    let negative = normalized.starts_with('-');
    if let Some(suffix) = normalized.strip_prefix('-') {
        if suffix.starts_with(['-', '+']) {
            return Err(original_error);
        }
        normalized = suffix;
    }
    let Some(mut normalized) = response_seed_leading_zero_suffix(normalized) else {
        return Err(original_error);
    };
    // The Source integer owner removes a zero-only fraction before parsing;
    // a floating conversion would round away significant integer digits.
    if let Some((integer, fraction)) = normalized.split_once('.')
        && !fraction.is_empty()
        && fraction.bytes().all(|byte| byte == b'0')
    {
        normalized = integer;
    }
    let normalized = if normalized.contains('_')
        && !normalized.starts_with('_')
        && !normalized.ends_with('_')
        && !normalized.contains("__")
    {
        std::borrow::Cow::Owned(normalized.replace('_', ""))
    } else {
        std::borrow::Cow::Borrowed(normalized)
    };
    if negative {
        format!("-{normalized}").parse()
    } else {
        normalized.parse()
    }
}
fn response_seed_leading_zero_suffix(value: &str) -> Option<&str> {
    let mut characters = value.char_indices();
    match characters.next()? {
        (_, '0') => {}
        (_, '1'..='9' | '-') => return Some(value),
        _ => return None,
    }
    for (index, character) in characters {
        match character {
            '0' | '_' => {}
            '1'..='9' | '-' => return value.get(index..),
            '.' => return value.get(index.checked_sub(1)?..),
            _ => return None,
        }
    }
    value.get(value.len().checked_sub(1)?..)
}
fn deserialize_response_created<'de, Deserializer>(
    deserializer: Deserializer,
) -> Result<Option<speedate::DateTime>, Deserializer::Error>
where
    Deserializer: serde::Deserializer<'de>,
{
    Ok(Option::<ResponseCreated>::deserialize(deserializer)?.map(|created| created.0))
}
pub(crate) struct ResponseCreated(speedate::DateTime);
impl<'de> Deserialize<'de> for ResponseCreated {
    fn deserialize<Deserializer>(deserializer: Deserializer) -> Result<Self, Deserializer::Error>
    where
        Deserializer: serde::Deserializer<'de>,
    {
        struct CreatedVisitor;
        impl<'de> serde::de::Visitor<'de> for CreatedVisitor {
            type Value = speedate::DateTime;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a datetime string or numeric timestamp")
            }
            fn visit_str<Error>(self, value: &str) -> Result<Self::Value, Error>
            where
                Error: serde::de::Error,
            {
                let config = speedate::DateTimeConfig {
                    time_config: speedate::TimeConfig {
                        microseconds_precision_overflow_behavior:
                            speedate::MicrosecondsPrecisionOverflowBehavior::Truncate,
                        unix_timestamp_offset: Some(0),
                    },
                    timestamp_unit: speedate::TimestampUnit::Infer,
                };
                let datetime =
                    speedate::DateTime::parse_bytes_with_config(value.as_bytes(), &config)
                        // The lax Source validator retries a datetime parse failure as a date.
                        .or_else(|_| {
                            speedate::Date::parse_bytes_with_config(
                                value.as_bytes(),
                                &speedate::DateConfig {
                                    timestamp_unit: speedate::TimestampUnit::Infer,
                                },
                            )
                            .map(|date| speedate::DateTime {
                                date,
                                time: speedate::Time {
                                    hour: 0,
                                    minute: 0,
                                    second: 0,
                                    microsecond: 0,
                                    tz_offset: None,
                                },
                            })
                        })
                        .map_err(Error::custom)?;
                checked_response_created(datetime)
            }
            fn visit_i64<Error>(self, value: i64) -> Result<Self::Value, Error>
            where
                Error: serde::de::Error,
            {
                let datetime = speedate::DateTime::from_timestamp_with_config(
                    value,
                    0,
                    &response_timestamp_config(),
                )
                .map_err(Error::custom)?;
                checked_response_created(datetime)
            }
            fn visit_u64<Error>(self, value: u64) -> Result<Self::Value, Error>
            where
                Error: serde::de::Error,
            {
                match i64::try_from(value) {
                    Ok(value) => self.visit_i64(value),
                    // Python falls back to float extraction when an integer exceeds i64.
                    Err(_) => self.visit_f64(value as f64),
                }
            }
            fn visit_f64<Error>(self, value: f64) -> Result<Self::Value, Error>
            where
                Error: serde::de::Error,
            {
                if value.is_nan() {
                    return Err(Error::custom("NaN values not permitted"));
                }
                let datetime =
                    speedate::DateTime::from_float_with_config(value, &response_timestamp_config())
                        .map_err(Error::custom)?;
                checked_response_created(datetime)
            }
        }
        deserializer.deserialize_any(CreatedVisitor).map(Self)
    }
}
fn response_timestamp_config() -> speedate::DateTimeConfig {
    speedate::DateTimeConfig {
        time_config: speedate::TimeConfig {
            unix_timestamp_offset: Some(0),
            ..Default::default()
        },
        timestamp_unit: speedate::TimestampUnit::Infer,
    }
}
fn checked_response_created<Error>(
    datetime: speedate::DateTime,
) -> Result<speedate::DateTime, Error>
where
    Error: serde::de::Error,
{
    if !(1..=9999).contains(&datetime.date.year) {
        return Err(Error::custom(format!(
            "year {} is out of range",
            datetime.date.year
        )));
    }
    Ok(datetime)
}
