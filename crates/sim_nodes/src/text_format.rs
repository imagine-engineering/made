use std::collections::BTreeMap;

use serde_json::Value;
use sim_types::CancellationToken;
use thiserror::Error;

use crate::{NativePrimitive, NativeValue};

pub const NATIVE_TEXT_FORMAT_MAX_TEMPLATE_BYTES: usize = 1024 * 1024;
pub const NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES: usize = 16 * 1024 * 1024;
const NATIVE_TEXT_FORMAT_MAX_DEPTH: usize = 64;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum NativeTextFormatError {
    #[error("text format template is {actual_bytes} bytes, above the {maximum_bytes}-byte limit")]
    TemplateTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    #[error("text format result exceeded the {maximum_bytes}-byte limit")]
    ResultTooLarge { maximum_bytes: usize },
    #[error("invalid text format template: {0}")]
    InvalidTemplate(String),
    #[error("text format field `{0}` is missing")]
    MissingField(String),
    #[error("text format field access is invalid: {0}")]
    InvalidFieldAccess(String),
    #[error("text format value is not safely representable: {0}")]
    UnsupportedValue(String),
    #[error("text formatting was cancelled")]
    Cancelled,
}

pub struct NativeTextFormatter;

impl NativeTextFormatter {
    pub const SEMANTIC_VERSION: &'static str = "source-bb019631-text-v4";

    pub fn format(
        template: &str,
        values: &BTreeMap<String, NativeValue>,
        cancellation: &CancellationToken,
    ) -> Result<String, NativeTextFormatError> {
        if template.len() > NATIVE_TEXT_FORMAT_MAX_TEMPLATE_BYTES {
            return Err(NativeTextFormatError::TemplateTooLarge {
                actual_bytes: template.len(),
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_TEMPLATE_BYTES,
            });
        }
        render_template(template, values, cancellation, 0)
    }
}

fn render_template(
    template: &str,
    values: &BTreeMap<String, NativeValue>,
    cancellation: &CancellationToken,
    depth: usize,
) -> Result<String, NativeTextFormatError> {
    check_cancellation(cancellation)?;
    if depth > NATIVE_TEXT_FORMAT_MAX_DEPTH {
        return Err(NativeTextFormatError::InvalidTemplate(
            "nested replacement fields are too deep".to_owned(),
        ));
    }
    let mut output = String::new();
    let mut position = 0usize;
    while position < template.len() {
        check_cancellation(cancellation)?;
        let remainder = template.get(position..).ok_or_else(invalid_utf8_boundary)?;
        let character = remainder.chars().next().ok_or_else(invalid_utf8_boundary)?;
        match character {
            '{' if remainder.starts_with("{{") => {
                append_bounded(&mut output, "{")?;
                position += 2;
            }
            '}' if remainder.starts_with("}}") => {
                append_bounded(&mut output, "}")?;
                position += 2;
            }
            '{' => {
                let (field, next_position) = replacement_field(template, position + 1)?;
                let parsed = ParsedField::checked(field)?;
                let resolved = resolve_field(parsed.field_name, values)?;
                let conversion = parsed.checked_conversion()?;
                let spec = if parsed.format_spec.is_empty() {
                    String::new()
                } else {
                    render_template(parsed.format_spec, values, cancellation, depth + 1)?
                };
                append_bounded(&mut output, &format_resolved(resolved, conversion, &spec)?)?;
                position = next_position;
            }
            '}' => {
                return Err(NativeTextFormatError::InvalidTemplate(
                    "single closing brace".to_owned(),
                ));
            }
            _ => {
                let mut encoded = [0u8; 4];
                append_bounded(&mut output, character.encode_utf8(&mut encoded))?;
                position += character.len_utf8();
            }
        }
    }
    check_cancellation(cancellation)?;
    Ok(output)
}

fn invalid_utf8_boundary() -> NativeTextFormatError {
    NativeTextFormatError::InvalidTemplate("template boundary was not valid UTF-8".to_owned())
}

fn replacement_field(template: &str, start: usize) -> Result<(&str, usize), NativeTextFormatError> {
    let mut nested = 0usize;
    let mut bracket_depth = 0usize;
    let mut in_format_spec = false;
    let mut position = start;
    while position < template.len() {
        let remainder = template.get(position..).ok_or_else(invalid_utf8_boundary)?;
        let character = remainder.chars().next().ok_or_else(invalid_utf8_boundary)?;
        if nested == 0 && !in_format_spec {
            match character {
                '[' => bracket_depth = bracket_depth.saturating_add(1),
                ']' if bracket_depth > 0 => bracket_depth -= 1,
                ':' if bracket_depth == 0 => in_format_spec = true,
                '!' if bracket_depth == 0 => {
                    position += character.len_utf8();
                    if let Some(conversion) = template
                        .get(position..)
                        .and_then(|value| value.chars().next())
                    {
                        position += conversion.len_utf8();
                        continue;
                    }
                }
                _ => {}
            }
        }
        match character {
            '{' => nested = nested.saturating_add(1),
            '}' if nested == 0 => {
                return Ok((
                    template
                        .get(start..position)
                        .ok_or_else(invalid_utf8_boundary)?,
                    position + 1,
                ));
            }
            '}' => nested -= 1,
            _ => {}
        }
        position += character.len_utf8();
    }
    Err(NativeTextFormatError::InvalidTemplate(
        "replacement field is unterminated".to_owned(),
    ))
}

struct ParsedField<'a> {
    field_name: &'a str,
    conversion: Option<char>,
    format_spec: &'a str,
}

impl<'a> ParsedField<'a> {
    fn checked(field: &'a str) -> Result<Self, NativeTextFormatError> {
        let mut bracket_depth = 0usize;
        let mut conversion = None;
        let mut field_end = None;
        let mut format = None;
        let mut characters = field.char_indices();
        while let Some((position, character)) = characters.next() {
            match character {
                '[' => bracket_depth = bracket_depth.saturating_add(1),
                ']' if bracket_depth > 0 => bracket_depth -= 1,
                '!' if bracket_depth == 0 => {
                    field_end = Some(position);
                    conversion = Some(characters.next().ok_or_else(invalid_conversion)?.1);
                    match characters.next() {
                        None => break,
                        Some((position, ':')) => {
                            format = Some(position);
                            break;
                        }
                        Some(_) => return Err(invalid_conversion()),
                    }
                }
                ':' if bracket_depth == 0 => {
                    format = Some(position);
                    break;
                }
                _ => {}
            }
        }
        let field_end = field_end.or(format).unwrap_or(field.len());
        let field_name = field.get(..field_end).unwrap_or_default();
        if field_name.is_empty() {
            return Err(NativeTextFormatError::InvalidTemplate(
                "keyword-only formatting requires a field name".to_owned(),
            ));
        }
        let format_spec = format
            .and_then(|position| field.get(position + 1..))
            .unwrap_or_default();
        Ok(Self {
            field_name,
            conversion,
            format_spec,
        })
    }

    fn checked_conversion(&self) -> Result<Option<char>, NativeTextFormatError> {
        match self.conversion {
            None | Some('s' | 'r' | 'a') => Ok(self.conversion),
            Some(_) => Err(invalid_conversion()),
        }
    }
}

fn invalid_conversion() -> NativeTextFormatError {
    NativeTextFormatError::InvalidTemplate("conversion must be exactly !s, !r, or !a".to_owned())
}

enum ValueRef<'a> {
    Native(&'a NativeValue),
    Json(&'a Value),
    Character(char),
    Scalar(FormatAtom),
}

fn resolve_field<'a>(
    field: &str,
    values: &'a BTreeMap<String, NativeValue>,
) -> Result<ValueRef<'a>, NativeTextFormatError> {
    let root_end = field.find(['.', '[']).unwrap_or(field.len());
    let root = field.get(..root_end).unwrap_or_default();
    if python_decimal_field_index(root, field)?.is_some() {
        return Err(NativeTextFormatError::InvalidFieldAccess(field.to_owned()));
    }
    let mut value = ValueRef::Native(
        values
            .get(root)
            .ok_or_else(|| NativeTextFormatError::MissingField(root.to_owned()))?,
    );
    let mut remainder = field.get(root_end..).unwrap_or_default();
    while !remainder.is_empty() {
        if let Some(attribute) = remainder.strip_prefix('.') {
            let end = attribute.find(['.', '[']).unwrap_or(attribute.len());
            value =
                lookup_scalar_attribute(value, attribute.get(..end).unwrap_or_default(), field)?;
            remainder = attribute.get(end..).unwrap_or_default();
        } else if let Some(indexed) = remainder.strip_prefix('[') {
            let end = indexed
                .find(']')
                .ok_or_else(|| NativeTextFormatError::InvalidFieldAccess(field.to_owned()))?;
            value = lookup_index(value, indexed.get(..end).unwrap_or_default(), field)?;
            remainder = indexed.get(end + 1..).unwrap_or_default();
        } else {
            return Err(NativeTextFormatError::InvalidFieldAccess(field.to_owned()));
        }
    }
    Ok(value)
}

fn lookup_scalar_attribute<'a>(
    value: ValueRef<'a>,
    name: &str,
    field: &str,
) -> Result<ValueRef<'a>, NativeTextFormatError> {
    let invalid = || NativeTextFormatError::InvalidFieldAccess(field.to_owned());
    let atom = match value {
        ValueRef::Scalar(atom) => atom,
        ValueRef::Native(
            value @ NativeValue::Primitive {
                value:
                    NativePrimitive::Boolean(_)
                    | NativePrimitive::Integer(_)
                    | NativePrimitive::UnsignedInteger(_)
                    | NativePrimitive::Number(_),
            },
        ) => default_atom(ValueRef::Native(value))?,
        ValueRef::Native(NativeValue::PreservedUnknown { value, .. }) | ValueRef::Json(value)
            if matches!(value, Value::Number(_) | Value::Bool(_)) =>
        {
            default_atom(ValueRef::Json(value))?
        }
        _ => return Err(invalid()),
    };
    let result = match (atom, name) {
        (FormatAtom::Signed(value), "real" | "numerator") => FormatAtom::Signed(value),
        (FormatAtom::Unsigned(value), "real" | "numerator") => FormatAtom::Unsigned(value),
        (FormatAtom::Boolean(value), "real" | "numerator") => FormatAtom::Signed(i64::from(value)),
        (FormatAtom::Float(value), "real") => FormatAtom::Float(value),
        (FormatAtom::Float(_), "imag") => FormatAtom::Float(0.0),
        (FormatAtom::Signed(_) | FormatAtom::Unsigned(_) | FormatAtom::Boolean(_), "imag") => {
            FormatAtom::Signed(0)
        }
        (
            FormatAtom::Signed(_) | FormatAtom::Unsigned(_) | FormatAtom::Boolean(_),
            "denominator",
        ) => FormatAtom::Signed(1),
        _ => return Err(invalid()),
    };
    Ok(ValueRef::Scalar(result))
}

fn lookup_index<'a>(
    value: ValueRef<'a>,
    key: &str,
    field: &str,
) -> Result<ValueRef<'a>, NativeTextFormatError> {
    let invalid = || NativeTextFormatError::InvalidFieldAccess(field.to_owned());
    let index = python_decimal_field_index(key, field)?;
    match value {
        ValueRef::Native(NativeValue::List { values }) => index
            .and_then(|index| values.get(index))
            .map(ValueRef::Native)
            .ok_or_else(invalid),
        ValueRef::Native(NativeValue::Primitive {
            value: NativePrimitive::String(value),
        }) => index
            .and_then(|index| value.chars().nth(index))
            .map(ValueRef::Character)
            .ok_or_else(invalid),
        ValueRef::Native(NativeValue::PreservedUnknown { value, .. }) | ValueRef::Json(value) => {
            match value {
                Value::Array(values) => index
                    .and_then(|index| values.get(index))
                    .map(ValueRef::Json)
                    .ok_or_else(invalid),
                Value::String(value) => index
                    .and_then(|index| value.chars().nth(index))
                    .map(ValueRef::Character)
                    .ok_or_else(invalid),
                Value::Object(values) if index.is_none() => {
                    values.get(key).map(ValueRef::Json).ok_or_else(invalid)
                }
                _ => Err(invalid()),
            }
        }
        _ => Err(invalid()),
    }
}

fn python_decimal_field_index(
    key: &str,
    field: &str,
) -> Result<Option<usize>, NativeTextFormatError> {
    let invalid = || NativeTextFormatError::InvalidFieldAccess(field.to_owned());
    if key.is_empty() {
        return Err(invalid());
    }
    let mut index = 0usize;
    for character in key.chars() {
        let Some(digit) = python_unicode_15_decimal_digit(character) else {
            return Ok(None);
        };
        index = index
            .checked_mul(10)
            .and_then(|value| value.checked_add(usize::from(digit)))
            .filter(|value| *value <= isize::MAX as usize)
            .ok_or_else(invalid)?;
    }
    Ok(Some(index))
}

fn python_unicode_15_decimal_digit(character: char) -> Option<u8> {
    let codepoint = u32::from(character);
    let index = PYTHON_UNICODE_15_DECIMAL_ZEROES.partition_point(|zero| *zero <= codepoint);
    let zero = PYTHON_UNICODE_15_DECIMAL_ZEROES.get(index.checked_sub(1)?)?;
    let offset = codepoint.checked_sub(*zero)?;
    if offset < 10 {
        u8::try_from(offset).ok()
    } else {
        None
    }
}

// Dependency and Rust Unicode releases can advance beyond the pinned Python 3.12 source.
const PYTHON_UNICODE_15_DECIMAL_ZEROES: &[u32] = &[
    0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66,
    0xde6, 0xe50, 0xed0, 0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90,
    0x1b50, 0x1bb0, 0x1c40, 0x1c50, 0xa620, 0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0, 0xff10,
    0x104a0, 0x10d30, 0x11066, 0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650,
    0x116c0, 0x11730, 0x118e0, 0x11950, 0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16a60, 0x16ac0,
    0x16b50, 0x1d7ce, 0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950,
    0x1fbf0,
];

enum FormatAtom {
    Text(String),
    Boolean(bool),
    Signed(i64),
    Unsigned(u64),
    Float(f64),
}

fn format_resolved(
    value: ValueRef<'_>,
    conversion: Option<char>,
    spec: &str,
) -> Result<String, NativeTextFormatError> {
    let atom = match conversion {
        Some('s') => FormatAtom::Text(python_string(value)?),
        Some('r') => FormatAtom::Text(python_repr(value, false)?),
        Some('a') => FormatAtom::Text(python_repr(value, true)?),
        None => default_atom(value)?,
        Some(_) => unreachable!(),
    };
    apply_spec(atom, spec)
}

fn default_atom(value: ValueRef<'_>) -> Result<FormatAtom, NativeTextFormatError> {
    match value {
        ValueRef::Scalar(atom) => Ok(atom),
        ValueRef::Native(NativeValue::Primitive { value }) => Ok(match value {
            NativePrimitive::Null => FormatAtom::Text("None".to_owned()),
            NativePrimitive::Boolean(value) => FormatAtom::Boolean(*value),
            NativePrimitive::Integer(value) => FormatAtom::Signed(*value),
            NativePrimitive::UnsignedInteger(value) => FormatAtom::Unsigned(*value),
            NativePrimitive::Number(value) => FormatAtom::Float(*value),
            NativePrimitive::String(value) => FormatAtom::Text(value.clone()),
        }),
        ValueRef::Native(NativeValue::PreservedUnknown { value, .. }) | ValueRef::Json(value) => {
            match value {
                Value::String(value) => Ok(FormatAtom::Text(value.clone())),
                Value::Number(value) => value
                    .as_i64()
                    .map(FormatAtom::Signed)
                    .or_else(|| value.as_u64().map(FormatAtom::Unsigned))
                    .or_else(|| value.as_f64().map(FormatAtom::Float))
                    .ok_or_else(|| {
                        NativeTextFormatError::UnsupportedValue("JSON number".to_owned())
                    }),
                Value::Bool(value) => Ok(FormatAtom::Boolean(*value)),
                Value::Null => Ok(FormatAtom::Text("None".to_owned())),
                Value::Array(_) | Value::Object(_) => {
                    Ok(FormatAtom::Text(json_repr(value, false)?))
                }
            }
        }
        value => Ok(FormatAtom::Text(python_string(value)?)),
    }
}

fn python_string(value: ValueRef<'_>) -> Result<String, NativeTextFormatError> {
    match value {
        ValueRef::Scalar(atom) => apply_spec(atom, ""),
        ValueRef::Character(value) => Ok(value.to_string()),
        ValueRef::Native(NativeValue::PreservedUnknown {
            value: Value::String(value),
            ..
        })
        | ValueRef::Json(Value::String(value)) => Ok(value.clone()),
        ValueRef::Native(NativeValue::Primitive { value }) => Ok(match value {
            NativePrimitive::Null => "None".to_owned(),
            NativePrimitive::Boolean(value) => if *value { "True" } else { "False" }.to_owned(),
            NativePrimitive::Integer(value) => value.to_string(),
            NativePrimitive::UnsignedInteger(value) => value.to_string(),
            NativePrimitive::Number(value) => python_float(*value),
            NativePrimitive::String(value) => value.clone(),
        }),
        ValueRef::Native(NativeValue::Handle { .. }) => Err(
            NativeTextFormatError::UnsupportedValue("process-local handle".to_owned()),
        ),
        value => python_repr(value, false),
    }
}

fn python_repr(value: ValueRef<'_>, ascii_only: bool) -> Result<String, NativeTextFormatError> {
    match value {
        ValueRef::Scalar(FormatAtom::Text(value)) => Ok(quoted(&value, ascii_only)),
        ValueRef::Scalar(atom) => apply_spec(atom, ""),
        ValueRef::Character(value) => Ok(quoted(&value.to_string(), ascii_only)),
        ValueRef::Native(NativeValue::Primitive { value }) => match value {
            NativePrimitive::String(value) => Ok(quoted(value, ascii_only)),
            NativePrimitive::Null => Ok("None".to_owned()),
            NativePrimitive::Boolean(value) => Ok(if *value { "True" } else { "False" }.to_owned()),
            NativePrimitive::Integer(value) => Ok(value.to_string()),
            NativePrimitive::UnsignedInteger(value) => Ok(value.to_string()),
            NativePrimitive::Number(value) => Ok(python_float(*value)),
        },
        ValueRef::Native(NativeValue::List { values }) => {
            let values = values
                .iter()
                .map(|value| python_repr(ValueRef::Native(value), ascii_only))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(format!("[{}]", values.join(", ")))
        }
        ValueRef::Native(NativeValue::PreservedUnknown { value, .. }) | ValueRef::Json(value) => {
            json_repr(value, ascii_only)
        }
        ValueRef::Native(NativeValue::Handle { .. }) => Err(
            NativeTextFormatError::UnsupportedValue("process-local handle".to_owned()),
        ),
    }
}

fn json_repr(value: &Value, ascii_only: bool) -> Result<String, NativeTextFormatError> {
    Ok(match value {
        Value::Null => "None".to_owned(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                value.to_string()
            } else if let Some(value) = value.as_u64() {
                value.to_string()
            } else if let Some(value) = value.as_f64() {
                python_float(value)
            } else {
                return Err(NativeTextFormatError::UnsupportedValue(
                    "JSON number".to_owned(),
                ));
            }
        }
        Value::String(value) => quoted(value, ascii_only),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(|value| json_repr(value, ascii_only))
                .collect::<Result<Vec<_>, _>>()?
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| Ok(format!(
                    "{}: {}",
                    quoted(key, ascii_only),
                    json_repr(value, ascii_only)?
                )))
                .collect::<Result<Vec<_>, NativeTextFormatError>>()?
                .join(", ")
        ),
    })
}

fn quoted(value: &str, ascii_only: bool) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut output = String::new();
    output.push(quote);
    for character in value.chars() {
        match character {
            '\\' => output.push_str("\\\\"),
            character if character == quote => {
                output.push('\\');
                output.push(character);
            }
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\u{0008}' => output.push_str("\\x08"),
            '\u{000c}' => output.push_str("\\x0c"),
            character
                if !python_unicode_15_printable(character)
                    || (ascii_only && !character.is_ascii()) =>
            {
                let value = u32::from(character);
                if value <= 0xff {
                    output.push_str(&format!("\\x{value:02x}"));
                } else if value <= 0xffff {
                    output.push_str(&format!("\\u{value:04x}"));
                } else {
                    output.push_str(&format!("\\U{value:08x}"));
                }
            }
            character => output.push(character),
        }
    }
    output.push(quote);
    output
}

fn python_unicode_15_printable(character: char) -> bool {
    let value = u32::from(character);
    let index = PYTHON_UNICODE_15_NONPRINTABLE_RANGES.partition_point(|(_, end)| *end < value);
    !PYTHON_UNICODE_15_NONPRINTABLE_RANGES
        .get(index)
        .is_some_and(|(start, _)| *start <= value)
}

// Rust and dependency Unicode versions can advance while the pinned source stays at Unicode 15.
const PYTHON_UNICODE_15_NONPRINTABLE_RANGES: &[(u32, u32)] = &[
    (0x0, 0x1f),
    (0x7f, 0xa0),
    (0xad, 0xad),
    (0x378, 0x379),
    (0x380, 0x383),
    (0x38b, 0x38b),
    (0x38d, 0x38d),
    (0x3a2, 0x3a2),
    (0x530, 0x530),
    (0x557, 0x558),
    (0x58b, 0x58c),
    (0x590, 0x590),
    (0x5c8, 0x5cf),
    (0x5eb, 0x5ee),
    (0x5f5, 0x605),
    (0x61c, 0x61c),
    (0x6dd, 0x6dd),
    (0x70e, 0x70f),
    (0x74b, 0x74c),
    (0x7b2, 0x7bf),
    (0x7fb, 0x7fc),
    (0x82e, 0x82f),
    (0x83f, 0x83f),
    (0x85c, 0x85d),
    (0x85f, 0x85f),
    (0x86b, 0x86f),
    (0x88f, 0x897),
    (0x8e2, 0x8e2),
    (0x984, 0x984),
    (0x98d, 0x98e),
    (0x991, 0x992),
    (0x9a9, 0x9a9),
    (0x9b1, 0x9b1),
    (0x9b3, 0x9b5),
    (0x9ba, 0x9bb),
    (0x9c5, 0x9c6),
    (0x9c9, 0x9ca),
    (0x9cf, 0x9d6),
    (0x9d8, 0x9db),
    (0x9de, 0x9de),
    (0x9e4, 0x9e5),
    (0x9ff, 0xa00),
    (0xa04, 0xa04),
    (0xa0b, 0xa0e),
    (0xa11, 0xa12),
    (0xa29, 0xa29),
    (0xa31, 0xa31),
    (0xa34, 0xa34),
    (0xa37, 0xa37),
    (0xa3a, 0xa3b),
    (0xa3d, 0xa3d),
    (0xa43, 0xa46),
    (0xa49, 0xa4a),
    (0xa4e, 0xa50),
    (0xa52, 0xa58),
    (0xa5d, 0xa5d),
    (0xa5f, 0xa65),
    (0xa77, 0xa80),
    (0xa84, 0xa84),
    (0xa8e, 0xa8e),
    (0xa92, 0xa92),
    (0xaa9, 0xaa9),
    (0xab1, 0xab1),
    (0xab4, 0xab4),
    (0xaba, 0xabb),
    (0xac6, 0xac6),
    (0xaca, 0xaca),
    (0xace, 0xacf),
    (0xad1, 0xadf),
    (0xae4, 0xae5),
    (0xaf2, 0xaf8),
    (0xb00, 0xb00),
    (0xb04, 0xb04),
    (0xb0d, 0xb0e),
    (0xb11, 0xb12),
    (0xb29, 0xb29),
    (0xb31, 0xb31),
    (0xb34, 0xb34),
    (0xb3a, 0xb3b),
    (0xb45, 0xb46),
    (0xb49, 0xb4a),
    (0xb4e, 0xb54),
    (0xb58, 0xb5b),
    (0xb5e, 0xb5e),
    (0xb64, 0xb65),
    (0xb78, 0xb81),
    (0xb84, 0xb84),
    (0xb8b, 0xb8d),
    (0xb91, 0xb91),
    (0xb96, 0xb98),
    (0xb9b, 0xb9b),
    (0xb9d, 0xb9d),
    (0xba0, 0xba2),
    (0xba5, 0xba7),
    (0xbab, 0xbad),
    (0xbba, 0xbbd),
    (0xbc3, 0xbc5),
    (0xbc9, 0xbc9),
    (0xbce, 0xbcf),
    (0xbd1, 0xbd6),
    (0xbd8, 0xbe5),
    (0xbfb, 0xbff),
    (0xc0d, 0xc0d),
    (0xc11, 0xc11),
    (0xc29, 0xc29),
    (0xc3a, 0xc3b),
    (0xc45, 0xc45),
    (0xc49, 0xc49),
    (0xc4e, 0xc54),
    (0xc57, 0xc57),
    (0xc5b, 0xc5c),
    (0xc5e, 0xc5f),
    (0xc64, 0xc65),
    (0xc70, 0xc76),
    (0xc8d, 0xc8d),
    (0xc91, 0xc91),
    (0xca9, 0xca9),
    (0xcb4, 0xcb4),
    (0xcba, 0xcbb),
    (0xcc5, 0xcc5),
    (0xcc9, 0xcc9),
    (0xcce, 0xcd4),
    (0xcd7, 0xcdc),
    (0xcdf, 0xcdf),
    (0xce4, 0xce5),
    (0xcf0, 0xcf0),
    (0xcf4, 0xcff),
    (0xd0d, 0xd0d),
    (0xd11, 0xd11),
    (0xd45, 0xd45),
    (0xd49, 0xd49),
    (0xd50, 0xd53),
    (0xd64, 0xd65),
    (0xd80, 0xd80),
    (0xd84, 0xd84),
    (0xd97, 0xd99),
    (0xdb2, 0xdb2),
    (0xdbc, 0xdbc),
    (0xdbe, 0xdbf),
    (0xdc7, 0xdc9),
    (0xdcb, 0xdce),
    (0xdd5, 0xdd5),
    (0xdd7, 0xdd7),
    (0xde0, 0xde5),
    (0xdf0, 0xdf1),
    (0xdf5, 0xe00),
    (0xe3b, 0xe3e),
    (0xe5c, 0xe80),
    (0xe83, 0xe83),
    (0xe85, 0xe85),
    (0xe8b, 0xe8b),
    (0xea4, 0xea4),
    (0xea6, 0xea6),
    (0xebe, 0xebf),
    (0xec5, 0xec5),
    (0xec7, 0xec7),
    (0xecf, 0xecf),
    (0xeda, 0xedb),
    (0xee0, 0xeff),
    (0xf48, 0xf48),
    (0xf6d, 0xf70),
    (0xf98, 0xf98),
    (0xfbd, 0xfbd),
    (0xfcd, 0xfcd),
    (0xfdb, 0xfff),
    (0x10c6, 0x10c6),
    (0x10c8, 0x10cc),
    (0x10ce, 0x10cf),
    (0x1249, 0x1249),
    (0x124e, 0x124f),
    (0x1257, 0x1257),
    (0x1259, 0x1259),
    (0x125e, 0x125f),
    (0x1289, 0x1289),
    (0x128e, 0x128f),
    (0x12b1, 0x12b1),
    (0x12b6, 0x12b7),
    (0x12bf, 0x12bf),
    (0x12c1, 0x12c1),
    (0x12c6, 0x12c7),
    (0x12d7, 0x12d7),
    (0x1311, 0x1311),
    (0x1316, 0x1317),
    (0x135b, 0x135c),
    (0x137d, 0x137f),
    (0x139a, 0x139f),
    (0x13f6, 0x13f7),
    (0x13fe, 0x13ff),
    (0x1680, 0x1680),
    (0x169d, 0x169f),
    (0x16f9, 0x16ff),
    (0x1716, 0x171e),
    (0x1737, 0x173f),
    (0x1754, 0x175f),
    (0x176d, 0x176d),
    (0x1771, 0x1771),
    (0x1774, 0x177f),
    (0x17de, 0x17df),
    (0x17ea, 0x17ef),
    (0x17fa, 0x17ff),
    (0x180e, 0x180e),
    (0x181a, 0x181f),
    (0x1879, 0x187f),
    (0x18ab, 0x18af),
    (0x18f6, 0x18ff),
    (0x191f, 0x191f),
    (0x192c, 0x192f),
    (0x193c, 0x193f),
    (0x1941, 0x1943),
    (0x196e, 0x196f),
    (0x1975, 0x197f),
    (0x19ac, 0x19af),
    (0x19ca, 0x19cf),
    (0x19db, 0x19dd),
    (0x1a1c, 0x1a1d),
    (0x1a5f, 0x1a5f),
    (0x1a7d, 0x1a7e),
    (0x1a8a, 0x1a8f),
    (0x1a9a, 0x1a9f),
    (0x1aae, 0x1aaf),
    (0x1acf, 0x1aff),
    (0x1b4d, 0x1b4f),
    (0x1b7f, 0x1b7f),
    (0x1bf4, 0x1bfb),
    (0x1c38, 0x1c3a),
    (0x1c4a, 0x1c4c),
    (0x1c89, 0x1c8f),
    (0x1cbb, 0x1cbc),
    (0x1cc8, 0x1ccf),
    (0x1cfb, 0x1cff),
    (0x1f16, 0x1f17),
    (0x1f1e, 0x1f1f),
    (0x1f46, 0x1f47),
    (0x1f4e, 0x1f4f),
    (0x1f58, 0x1f58),
    (0x1f5a, 0x1f5a),
    (0x1f5c, 0x1f5c),
    (0x1f5e, 0x1f5e),
    (0x1f7e, 0x1f7f),
    (0x1fb5, 0x1fb5),
    (0x1fc5, 0x1fc5),
    (0x1fd4, 0x1fd5),
    (0x1fdc, 0x1fdc),
    (0x1ff0, 0x1ff1),
    (0x1ff5, 0x1ff5),
    (0x1fff, 0x200f),
    (0x2028, 0x202f),
    (0x205f, 0x206f),
    (0x2072, 0x2073),
    (0x208f, 0x208f),
    (0x209d, 0x209f),
    (0x20c1, 0x20cf),
    (0x20f1, 0x20ff),
    (0x218c, 0x218f),
    (0x2427, 0x243f),
    (0x244b, 0x245f),
    (0x2b74, 0x2b75),
    (0x2b96, 0x2b96),
    (0x2cf4, 0x2cf8),
    (0x2d26, 0x2d26),
    (0x2d28, 0x2d2c),
    (0x2d2e, 0x2d2f),
    (0x2d68, 0x2d6e),
    (0x2d71, 0x2d7e),
    (0x2d97, 0x2d9f),
    (0x2da7, 0x2da7),
    (0x2daf, 0x2daf),
    (0x2db7, 0x2db7),
    (0x2dbf, 0x2dbf),
    (0x2dc7, 0x2dc7),
    (0x2dcf, 0x2dcf),
    (0x2dd7, 0x2dd7),
    (0x2ddf, 0x2ddf),
    (0x2e5e, 0x2e7f),
    (0x2e9a, 0x2e9a),
    (0x2ef4, 0x2eff),
    (0x2fd6, 0x2fef),
    (0x2ffc, 0x3000),
    (0x3040, 0x3040),
    (0x3097, 0x3098),
    (0x3100, 0x3104),
    (0x3130, 0x3130),
    (0x318f, 0x318f),
    (0x31e4, 0x31ef),
    (0x321f, 0x321f),
    (0xa48d, 0xa48f),
    (0xa4c7, 0xa4cf),
    (0xa62c, 0xa63f),
    (0xa6f8, 0xa6ff),
    (0xa7cb, 0xa7cf),
    (0xa7d2, 0xa7d2),
    (0xa7d4, 0xa7d4),
    (0xa7da, 0xa7f1),
    (0xa82d, 0xa82f),
    (0xa83a, 0xa83f),
    (0xa878, 0xa87f),
    (0xa8c6, 0xa8cd),
    (0xa8da, 0xa8df),
    (0xa954, 0xa95e),
    (0xa97d, 0xa97f),
    (0xa9ce, 0xa9ce),
    (0xa9da, 0xa9dd),
    (0xa9ff, 0xa9ff),
    (0xaa37, 0xaa3f),
    (0xaa4e, 0xaa4f),
    (0xaa5a, 0xaa5b),
    (0xaac3, 0xaada),
    (0xaaf7, 0xab00),
    (0xab07, 0xab08),
    (0xab0f, 0xab10),
    (0xab17, 0xab1f),
    (0xab27, 0xab27),
    (0xab2f, 0xab2f),
    (0xab6c, 0xab6f),
    (0xabee, 0xabef),
    (0xabfa, 0xabff),
    (0xd7a4, 0xd7af),
    (0xd7c7, 0xd7ca),
    (0xd7fc, 0xd7ff),
    (0xe000, 0xf8ff),
    (0xfa6e, 0xfa6f),
    (0xfada, 0xfaff),
    (0xfb07, 0xfb12),
    (0xfb18, 0xfb1c),
    (0xfb37, 0xfb37),
    (0xfb3d, 0xfb3d),
    (0xfb3f, 0xfb3f),
    (0xfb42, 0xfb42),
    (0xfb45, 0xfb45),
    (0xfbc3, 0xfbd2),
    (0xfd90, 0xfd91),
    (0xfdc8, 0xfdce),
    (0xfdd0, 0xfdef),
    (0xfe1a, 0xfe1f),
    (0xfe53, 0xfe53),
    (0xfe67, 0xfe67),
    (0xfe6c, 0xfe6f),
    (0xfe75, 0xfe75),
    (0xfefd, 0xff00),
    (0xffbf, 0xffc1),
    (0xffc8, 0xffc9),
    (0xffd0, 0xffd1),
    (0xffd8, 0xffd9),
    (0xffdd, 0xffdf),
    (0xffe7, 0xffe7),
    (0xffef, 0xfffb),
    (0xfffe, 0xffff),
    (0x1000c, 0x1000c),
    (0x10027, 0x10027),
    (0x1003b, 0x1003b),
    (0x1003e, 0x1003e),
    (0x1004e, 0x1004f),
    (0x1005e, 0x1007f),
    (0x100fb, 0x100ff),
    (0x10103, 0x10106),
    (0x10134, 0x10136),
    (0x1018f, 0x1018f),
    (0x1019d, 0x1019f),
    (0x101a1, 0x101cf),
    (0x101fe, 0x1027f),
    (0x1029d, 0x1029f),
    (0x102d1, 0x102df),
    (0x102fc, 0x102ff),
    (0x10324, 0x1032c),
    (0x1034b, 0x1034f),
    (0x1037b, 0x1037f),
    (0x1039e, 0x1039e),
    (0x103c4, 0x103c7),
    (0x103d6, 0x103ff),
    (0x1049e, 0x1049f),
    (0x104aa, 0x104af),
    (0x104d4, 0x104d7),
    (0x104fc, 0x104ff),
    (0x10528, 0x1052f),
    (0x10564, 0x1056e),
    (0x1057b, 0x1057b),
    (0x1058b, 0x1058b),
    (0x10593, 0x10593),
    (0x10596, 0x10596),
    (0x105a2, 0x105a2),
    (0x105b2, 0x105b2),
    (0x105ba, 0x105ba),
    (0x105bd, 0x105ff),
    (0x10737, 0x1073f),
    (0x10756, 0x1075f),
    (0x10768, 0x1077f),
    (0x10786, 0x10786),
    (0x107b1, 0x107b1),
    (0x107bb, 0x107ff),
    (0x10806, 0x10807),
    (0x10809, 0x10809),
    (0x10836, 0x10836),
    (0x10839, 0x1083b),
    (0x1083d, 0x1083e),
    (0x10856, 0x10856),
    (0x1089f, 0x108a6),
    (0x108b0, 0x108df),
    (0x108f3, 0x108f3),
    (0x108f6, 0x108fa),
    (0x1091c, 0x1091e),
    (0x1093a, 0x1093e),
    (0x10940, 0x1097f),
    (0x109b8, 0x109bb),
    (0x109d0, 0x109d1),
    (0x10a04, 0x10a04),
    (0x10a07, 0x10a0b),
    (0x10a14, 0x10a14),
    (0x10a18, 0x10a18),
    (0x10a36, 0x10a37),
    (0x10a3b, 0x10a3e),
    (0x10a49, 0x10a4f),
    (0x10a59, 0x10a5f),
    (0x10aa0, 0x10abf),
    (0x10ae7, 0x10aea),
    (0x10af7, 0x10aff),
    (0x10b36, 0x10b38),
    (0x10b56, 0x10b57),
    (0x10b73, 0x10b77),
    (0x10b92, 0x10b98),
    (0x10b9d, 0x10ba8),
    (0x10bb0, 0x10bff),
    (0x10c49, 0x10c7f),
    (0x10cb3, 0x10cbf),
    (0x10cf3, 0x10cf9),
    (0x10d28, 0x10d2f),
    (0x10d3a, 0x10e5f),
    (0x10e7f, 0x10e7f),
    (0x10eaa, 0x10eaa),
    (0x10eae, 0x10eaf),
    (0x10eb2, 0x10efc),
    (0x10f28, 0x10f2f),
    (0x10f5a, 0x10f6f),
    (0x10f8a, 0x10faf),
    (0x10fcc, 0x10fdf),
    (0x10ff7, 0x10fff),
    (0x1104e, 0x11051),
    (0x11076, 0x1107e),
    (0x110bd, 0x110bd),
    (0x110c3, 0x110cf),
    (0x110e9, 0x110ef),
    (0x110fa, 0x110ff),
    (0x11135, 0x11135),
    (0x11148, 0x1114f),
    (0x11177, 0x1117f),
    (0x111e0, 0x111e0),
    (0x111f5, 0x111ff),
    (0x11212, 0x11212),
    (0x11242, 0x1127f),
    (0x11287, 0x11287),
    (0x11289, 0x11289),
    (0x1128e, 0x1128e),
    (0x1129e, 0x1129e),
    (0x112aa, 0x112af),
    (0x112eb, 0x112ef),
    (0x112fa, 0x112ff),
    (0x11304, 0x11304),
    (0x1130d, 0x1130e),
    (0x11311, 0x11312),
    (0x11329, 0x11329),
    (0x11331, 0x11331),
    (0x11334, 0x11334),
    (0x1133a, 0x1133a),
    (0x11345, 0x11346),
    (0x11349, 0x1134a),
    (0x1134e, 0x1134f),
    (0x11351, 0x11356),
    (0x11358, 0x1135c),
    (0x11364, 0x11365),
    (0x1136d, 0x1136f),
    (0x11375, 0x113ff),
    (0x1145c, 0x1145c),
    (0x11462, 0x1147f),
    (0x114c8, 0x114cf),
    (0x114da, 0x1157f),
    (0x115b6, 0x115b7),
    (0x115de, 0x115ff),
    (0x11645, 0x1164f),
    (0x1165a, 0x1165f),
    (0x1166d, 0x1167f),
    (0x116ba, 0x116bf),
    (0x116ca, 0x116ff),
    (0x1171b, 0x1171c),
    (0x1172c, 0x1172f),
    (0x11747, 0x117ff),
    (0x1183c, 0x1189f),
    (0x118f3, 0x118fe),
    (0x11907, 0x11908),
    (0x1190a, 0x1190b),
    (0x11914, 0x11914),
    (0x11917, 0x11917),
    (0x11936, 0x11936),
    (0x11939, 0x1193a),
    (0x11947, 0x1194f),
    (0x1195a, 0x1199f),
    (0x119a8, 0x119a9),
    (0x119d8, 0x119d9),
    (0x119e5, 0x119ff),
    (0x11a48, 0x11a4f),
    (0x11aa3, 0x11aaf),
    (0x11af9, 0x11aff),
    (0x11b0a, 0x11bff),
    (0x11c09, 0x11c09),
    (0x11c37, 0x11c37),
    (0x11c46, 0x11c4f),
    (0x11c6d, 0x11c6f),
    (0x11c90, 0x11c91),
    (0x11ca8, 0x11ca8),
    (0x11cb7, 0x11cff),
    (0x11d07, 0x11d07),
    (0x11d0a, 0x11d0a),
    (0x11d37, 0x11d39),
    (0x11d3b, 0x11d3b),
    (0x11d3e, 0x11d3e),
    (0x11d48, 0x11d4f),
    (0x11d5a, 0x11d5f),
    (0x11d66, 0x11d66),
    (0x11d69, 0x11d69),
    (0x11d8f, 0x11d8f),
    (0x11d92, 0x11d92),
    (0x11d99, 0x11d9f),
    (0x11daa, 0x11edf),
    (0x11ef9, 0x11eff),
    (0x11f11, 0x11f11),
    (0x11f3b, 0x11f3d),
    (0x11f5a, 0x11faf),
    (0x11fb1, 0x11fbf),
    (0x11ff2, 0x11ffe),
    (0x1239a, 0x123ff),
    (0x1246f, 0x1246f),
    (0x12475, 0x1247f),
    (0x12544, 0x12f8f),
    (0x12ff3, 0x12fff),
    (0x13430, 0x1343f),
    (0x13456, 0x143ff),
    (0x14647, 0x167ff),
    (0x16a39, 0x16a3f),
    (0x16a5f, 0x16a5f),
    (0x16a6a, 0x16a6d),
    (0x16abf, 0x16abf),
    (0x16aca, 0x16acf),
    (0x16aee, 0x16aef),
    (0x16af6, 0x16aff),
    (0x16b46, 0x16b4f),
    (0x16b5a, 0x16b5a),
    (0x16b62, 0x16b62),
    (0x16b78, 0x16b7c),
    (0x16b90, 0x16e3f),
    (0x16e9b, 0x16eff),
    (0x16f4b, 0x16f4e),
    (0x16f88, 0x16f8e),
    (0x16fa0, 0x16fdf),
    (0x16fe5, 0x16fef),
    (0x16ff2, 0x16fff),
    (0x187f8, 0x187ff),
    (0x18cd6, 0x18cff),
    (0x18d09, 0x1afef),
    (0x1aff4, 0x1aff4),
    (0x1affc, 0x1affc),
    (0x1afff, 0x1afff),
    (0x1b123, 0x1b131),
    (0x1b133, 0x1b14f),
    (0x1b153, 0x1b154),
    (0x1b156, 0x1b163),
    (0x1b168, 0x1b16f),
    (0x1b2fc, 0x1bbff),
    (0x1bc6b, 0x1bc6f),
    (0x1bc7d, 0x1bc7f),
    (0x1bc89, 0x1bc8f),
    (0x1bc9a, 0x1bc9b),
    (0x1bca0, 0x1ceff),
    (0x1cf2e, 0x1cf2f),
    (0x1cf47, 0x1cf4f),
    (0x1cfc4, 0x1cfff),
    (0x1d0f6, 0x1d0ff),
    (0x1d127, 0x1d128),
    (0x1d173, 0x1d17a),
    (0x1d1eb, 0x1d1ff),
    (0x1d246, 0x1d2bf),
    (0x1d2d4, 0x1d2df),
    (0x1d2f4, 0x1d2ff),
    (0x1d357, 0x1d35f),
    (0x1d379, 0x1d3ff),
    (0x1d455, 0x1d455),
    (0x1d49d, 0x1d49d),
    (0x1d4a0, 0x1d4a1),
    (0x1d4a3, 0x1d4a4),
    (0x1d4a7, 0x1d4a8),
    (0x1d4ad, 0x1d4ad),
    (0x1d4ba, 0x1d4ba),
    (0x1d4bc, 0x1d4bc),
    (0x1d4c4, 0x1d4c4),
    (0x1d506, 0x1d506),
    (0x1d50b, 0x1d50c),
    (0x1d515, 0x1d515),
    (0x1d51d, 0x1d51d),
    (0x1d53a, 0x1d53a),
    (0x1d53f, 0x1d53f),
    (0x1d545, 0x1d545),
    (0x1d547, 0x1d549),
    (0x1d551, 0x1d551),
    (0x1d6a6, 0x1d6a7),
    (0x1d7cc, 0x1d7cd),
    (0x1da8c, 0x1da9a),
    (0x1daa0, 0x1daa0),
    (0x1dab0, 0x1deff),
    (0x1df1f, 0x1df24),
    (0x1df2b, 0x1dfff),
    (0x1e007, 0x1e007),
    (0x1e019, 0x1e01a),
    (0x1e022, 0x1e022),
    (0x1e025, 0x1e025),
    (0x1e02b, 0x1e02f),
    (0x1e06e, 0x1e08e),
    (0x1e090, 0x1e0ff),
    (0x1e12d, 0x1e12f),
    (0x1e13e, 0x1e13f),
    (0x1e14a, 0x1e14d),
    (0x1e150, 0x1e28f),
    (0x1e2af, 0x1e2bf),
    (0x1e2fa, 0x1e2fe),
    (0x1e300, 0x1e4cf),
    (0x1e4fa, 0x1e7df),
    (0x1e7e7, 0x1e7e7),
    (0x1e7ec, 0x1e7ec),
    (0x1e7ef, 0x1e7ef),
    (0x1e7ff, 0x1e7ff),
    (0x1e8c5, 0x1e8c6),
    (0x1e8d7, 0x1e8ff),
    (0x1e94c, 0x1e94f),
    (0x1e95a, 0x1e95d),
    (0x1e960, 0x1ec70),
    (0x1ecb5, 0x1ed00),
    (0x1ed3e, 0x1edff),
    (0x1ee04, 0x1ee04),
    (0x1ee20, 0x1ee20),
    (0x1ee23, 0x1ee23),
    (0x1ee25, 0x1ee26),
    (0x1ee28, 0x1ee28),
    (0x1ee33, 0x1ee33),
    (0x1ee38, 0x1ee38),
    (0x1ee3a, 0x1ee3a),
    (0x1ee3c, 0x1ee41),
    (0x1ee43, 0x1ee46),
    (0x1ee48, 0x1ee48),
    (0x1ee4a, 0x1ee4a),
    (0x1ee4c, 0x1ee4c),
    (0x1ee50, 0x1ee50),
    (0x1ee53, 0x1ee53),
    (0x1ee55, 0x1ee56),
    (0x1ee58, 0x1ee58),
    (0x1ee5a, 0x1ee5a),
    (0x1ee5c, 0x1ee5c),
    (0x1ee5e, 0x1ee5e),
    (0x1ee60, 0x1ee60),
    (0x1ee63, 0x1ee63),
    (0x1ee65, 0x1ee66),
    (0x1ee6b, 0x1ee6b),
    (0x1ee73, 0x1ee73),
    (0x1ee78, 0x1ee78),
    (0x1ee7d, 0x1ee7d),
    (0x1ee7f, 0x1ee7f),
    (0x1ee8a, 0x1ee8a),
    (0x1ee9c, 0x1eea0),
    (0x1eea4, 0x1eea4),
    (0x1eeaa, 0x1eeaa),
    (0x1eebc, 0x1eeef),
    (0x1eef2, 0x1efff),
    (0x1f02c, 0x1f02f),
    (0x1f094, 0x1f09f),
    (0x1f0af, 0x1f0b0),
    (0x1f0c0, 0x1f0c0),
    (0x1f0d0, 0x1f0d0),
    (0x1f0f6, 0x1f0ff),
    (0x1f1ae, 0x1f1e5),
    (0x1f203, 0x1f20f),
    (0x1f23c, 0x1f23f),
    (0x1f249, 0x1f24f),
    (0x1f252, 0x1f25f),
    (0x1f266, 0x1f2ff),
    (0x1f6d8, 0x1f6db),
    (0x1f6ed, 0x1f6ef),
    (0x1f6fd, 0x1f6ff),
    (0x1f777, 0x1f77a),
    (0x1f7da, 0x1f7df),
    (0x1f7ec, 0x1f7ef),
    (0x1f7f1, 0x1f7ff),
    (0x1f80c, 0x1f80f),
    (0x1f848, 0x1f84f),
    (0x1f85a, 0x1f85f),
    (0x1f888, 0x1f88f),
    (0x1f8ae, 0x1f8af),
    (0x1f8b2, 0x1f8ff),
    (0x1fa54, 0x1fa5f),
    (0x1fa6e, 0x1fa6f),
    (0x1fa7d, 0x1fa7f),
    (0x1fa89, 0x1fa8f),
    (0x1fabe, 0x1fabe),
    (0x1fac6, 0x1facd),
    (0x1fadc, 0x1fadf),
    (0x1fae9, 0x1faef),
    (0x1faf9, 0x1faff),
    (0x1fb93, 0x1fb93),
    (0x1fbcb, 0x1fbef),
    (0x1fbfa, 0x1ffff),
    (0x2a6e0, 0x2a6ff),
    (0x2b73a, 0x2b73f),
    (0x2b81e, 0x2b81f),
    (0x2cea2, 0x2ceaf),
    (0x2ebe1, 0x2f7ff),
    (0x2fa1e, 0x2ffff),
    (0x3134b, 0x3134f),
    (0x323b0, 0xe00ff),
    (0xe01f0, 0x10ffff),
];

fn python_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-inf"
        } else {
            "inf"
        }
        .to_owned();
    }
    let sign = if value.is_sign_negative() { "-" } else { "" };
    if value == 0.0 {
        return format!("{sign}0.0");
    }
    // The JSON serializer chooses the nearest even shortest decimal; Rust Display
    // instead rounds some equal-distance final digits away from zero.
    let spelling = Value::from(value.abs()).to_string();
    let (decimal, explicit_exponent) = match spelling.split_once('e') {
        Some((decimal, exponent)) => {
            let magnitude = exponent
                .bytes()
                .filter(u8::is_ascii_digit)
                .fold(0_i32, |value, digit| value * 10 + i32::from(digit - b'0'));
            (
                decimal,
                if exponent.starts_with('-') {
                    -magnitude
                } else {
                    magnitude
                },
            )
        }
        None => (spelling.as_str(), 0),
    };
    let (integer, fraction) = decimal.split_once('.').unwrap_or((decimal, ""));
    // Decimal digits preserve the threshold side even when log10 rounds an adjacent float.
    let exponent = explicit_exponent
        + if integer != "0" {
            integer.len() as i32 - 1
        } else {
            -(fraction
                .chars()
                .take_while(|character| *character == '0')
                .count() as i32)
                - 1
        };
    let digits = decimal
        .chars()
        .filter(|character| character.is_ascii_digit())
        .skip_while(|character| *character == '0')
        .collect::<String>();
    let digits = digits.trim_end_matches('0');
    if !(-4..16).contains(&exponent) {
        let mut coefficient = String::new();
        for (index, character) in digits.chars().enumerate() {
            if index == 1 {
                coefficient.push('.');
            }
            coefficient.push(character);
        }
        format!("{sign}{coefficient}e{exponent:+03}")
    } else if exponent < 0 {
        format!("{sign}0.{}{digits}", "0".repeat((-exponent - 1) as usize))
    } else {
        let decimal_position = (exponent + 1) as usize;
        if decimal_position >= digits.len() {
            format!(
                "{sign}{digits}{}.0",
                "0".repeat(decimal_position - digits.len())
            )
        } else {
            let mut decimal = String::new();
            for (index, character) in digits.chars().enumerate() {
                if index == decimal_position {
                    decimal.push('.');
                }
                decimal.push(character);
            }
            format!("{sign}{decimal}")
        }
    }
}

struct FormatSpec {
    fill: char,
    align: Option<char>,
    sign: Option<char>,
    coerce_negative_zero: bool,
    alternate: bool,
    zero: bool,
    width: Option<usize>,
    grouping: Option<char>,
    precision: Option<usize>,
    kind: Option<char>,
}

impl FormatSpec {
    fn checked(value: &str) -> Result<Self, NativeTextFormatError> {
        let values = value.chars().collect::<Vec<_>>();
        let mut position = 0usize;
        let explicit_fill = values.get(1).is_some_and(|value| "<>=^".contains(*value));
        let (fill, align) = if explicit_fill {
            position = 2;
            (values[0], values.get(1).copied())
        } else if values.first().is_some_and(|value| "<>=^".contains(*value)) {
            position = 1;
            (' ', values.first().copied())
        } else {
            (' ', None)
        };
        let sign = values
            .get(position)
            .copied()
            .filter(|value| "+- ".contains(*value));
        position += usize::from(sign.is_some());
        let coerce_negative_zero = values.get(position) == Some(&'z');
        position += usize::from(coerce_negative_zero);
        let alternate = values.get(position) == Some(&'#');
        position += usize::from(alternate);
        let zero = values.get(position) == Some(&'0');
        let fill = if zero && !explicit_fill { '0' } else { fill };
        position += usize::from(zero);
        let width_start = position;
        while values
            .get(position)
            .is_some_and(|value| python_unicode_15_decimal_digit(*value).is_some())
        {
            position += 1;
        }
        let width = parse_digits(&values[width_start..position])?;
        let grouping = values
            .get(position)
            .copied()
            .filter(|value| matches!(value, ',' | '_'));
        position += usize::from(grouping.is_some());
        if matches!(
            (grouping, values.get(position).copied()),
            (Some(','), Some('_')) | (Some('_'), Some(','))
        ) {
            return Err(NativeTextFormatError::InvalidTemplate(
                "cannot combine comma and underscore grouping".to_owned(),
            ));
        }
        let precision = if values.get(position) == Some(&'.') {
            position += 1;
            let start = position;
            while values
                .get(position)
                .is_some_and(|value| python_unicode_15_decimal_digit(*value).is_some())
            {
                position += 1;
            }
            Some(parse_digits(&values[start..position])?.ok_or_else(|| {
                NativeTextFormatError::InvalidTemplate("precision requires digits".to_owned())
            })?)
        } else {
            None
        };
        let kind = values.get(position).copied();
        position += usize::from(kind.is_some());
        if position != values.len() {
            return Err(NativeTextFormatError::InvalidTemplate(
                "unsupported format spec".to_owned(),
            ));
        }
        let grouping_supported = matches!(
            (grouping, kind),
            (None, _)
                | (
                    Some(_),
                    None | Some('d' | 'f' | 'F' | 'e' | 'E' | 'g' | 'G' | '%')
                )
                | (Some('_'), Some('b' | 'o' | 'x' | 'X'))
        );
        if !grouping_supported {
            return Err(NativeTextFormatError::InvalidTemplate(
                "grouping is unsupported for this format type".to_owned(),
            ));
        }
        Ok(Self {
            fill,
            align,
            sign,
            coerce_negative_zero,
            alternate,
            zero,
            width,
            grouping,
            precision,
            kind,
        })
    }
}

fn parse_digits(values: &[char]) -> Result<Option<usize>, NativeTextFormatError> {
    if values.is_empty() {
        return Ok(None);
    }
    let invalid =
        || NativeTextFormatError::InvalidTemplate("format number is too large".to_owned());
    let mut number = 0usize;
    for character in values {
        let digit = python_unicode_15_decimal_digit(*character).ok_or_else(invalid)?;
        number = number
            .checked_mul(10)
            .and_then(|value| value.checked_add(usize::from(digit)))
            .filter(|value| *value <= isize::MAX as usize)
            .ok_or_else(invalid)?;
    }
    Ok(Some(number))
}

fn apply_spec(value: FormatAtom, spec: &str) -> Result<String, NativeTextFormatError> {
    if spec.is_empty() {
        return Ok(match value {
            FormatAtom::Text(value) => value,
            FormatAtom::Boolean(value) => if value { "True" } else { "False" }.to_owned(),
            FormatAtom::Signed(value) => value.to_string(),
            FormatAtom::Unsigned(value) => value.to_string(),
            FormatAtom::Float(value) => python_float(value),
        });
    }
    let spec = FormatSpec::checked(spec)?;
    let numeric = !matches!(value, FormatAtom::Text(_));
    let mut value = match value {
        FormatAtom::Text(value) => format_text(value, &spec)?,
        FormatAtom::Boolean(value) => format_integer(u64::from(value), false, &spec)?,
        FormatAtom::Signed(value) => format_integer(value.unsigned_abs(), value < 0, &spec)?,
        FormatAtom::Unsigned(value) => format_integer(value, false, &spec)?,
        FormatAtom::Float(value) => format_number(value, &spec)?,
    };
    if let Some(width) = spec.width {
        let length = value.chars().count();
        if length < width {
            let padding = width - length;
            let align = spec.align.unwrap_or(if spec.zero && numeric {
                '='
            } else if numeric {
                '>'
            } else {
                '<'
            });
            // A formatted codepoint is not a numeric sign, even if it is '+' or '-'.
            let align = if numeric && spec.kind == Some('c') && align == '=' {
                '>'
            } else {
                align
            };
            let left = match align {
                '>' => padding,
                '^' => padding / 2,
                _ => 0,
            };
            let right = padding - left;
            let fill = spec.fill;
            if align == '=' {
                value = pad_numeric_after_prefix(value, fill, padding)?;
            } else {
                value = pad_value(value, fill, left, right)?;
            }
        }
    }
    if value.len() > NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES {
        return Err(NativeTextFormatError::ResultTooLarge {
            maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES,
        });
    }
    Ok(value)
}

fn format_text(mut value: String, spec: &FormatSpec) -> Result<String, NativeTextFormatError> {
    if spec.sign.is_some()
        || spec.coerce_negative_zero
        || spec.alternate
        || spec.grouping.is_some()
        || spec.align == Some('=')
        || !matches!(spec.kind, None | Some('s'))
    {
        return Err(NativeTextFormatError::InvalidTemplate(
            "numeric format applied to text".to_owned(),
        ));
    }
    if let Some(precision) = spec.precision {
        value = value.chars().take(precision).collect();
    }
    Ok(value)
}

fn format_integer(
    magnitude: u64,
    negative: bool,
    spec: &FormatSpec,
) -> Result<String, NativeTextFormatError> {
    if matches!(spec.kind, Some('f' | 'F' | 'e' | 'E' | 'g' | 'G' | '%')) {
        // Python converts these integer presentations to binary64 before applying float options.
        let value = magnitude as f64;
        let value = if negative { -value } else { value };
        return format_number(value, spec);
    }
    if spec.grouping.is_none()
        && !matches!(
            spec.kind,
            None | Some(
                'd' | 'n' | 'b' | 'o' | 'x' | 'X' | 'c' | 'f' | 'F' | 'e' | 'E' | 'g' | 'G' | '%'
            )
        )
    {
        return Err(NativeTextFormatError::InvalidTemplate(
            "unsupported integer format type".to_owned(),
        ));
    }
    if spec.precision.is_some() {
        return Err(NativeTextFormatError::InvalidTemplate(
            "integer precision is unsupported".to_owned(),
        ));
    }
    if spec.coerce_negative_zero {
        return Err(NativeTextFormatError::InvalidTemplate(
            "negative-zero coercion requires a floating-point value".to_owned(),
        ));
    }
    if spec.kind == Some('c') {
        if spec.sign.is_some() || spec.alternate || spec.grouping.is_some() {
            return Err(NativeTextFormatError::InvalidTemplate(
                "unsupported character format option".to_owned(),
            ));
        }
        let codepoint = u32::try_from(magnitude)
            .ok()
            .filter(|_| !negative)
            .and_then(char::from_u32)
            .ok_or_else(|| {
                NativeTextFormatError::InvalidTemplate(
                    "character code point is outside the Unicode scalar range".to_owned(),
                )
            })?;
        return Ok(codepoint.to_string());
    }
    let (prefix, mut digits, group_size) = match spec.kind.unwrap_or('d') {
        'd' | 'n' => ("", magnitude.to_string(), 3),
        'b' => (
            if spec.alternate { "0b" } else { "" },
            format!("{magnitude:b}"),
            4,
        ),
        'o' => (
            if spec.alternate { "0o" } else { "" },
            format!("{magnitude:o}"),
            4,
        ),
        'x' => (
            if spec.alternate { "0x" } else { "" },
            format!("{magnitude:x}"),
            4,
        ),
        'X' => (
            if spec.alternate { "0X" } else { "" },
            format!("{magnitude:X}"),
            4,
        ),
        _ => {
            return Err(NativeTextFormatError::InvalidTemplate(
                "unsupported integer format type".to_owned(),
            ));
        }
    };
    if let Some(grouping) = spec.grouping {
        if grouping == ',' && !matches!(spec.kind, None | Some('d' | 'n')) {
            return Err(NativeTextFormatError::InvalidTemplate(
                "comma grouping is unsupported for this integer format type".to_owned(),
            ));
        }
        let surrounding_bytes = sign_prefix(negative, spec.sign).len() + prefix.len();
        digits = group_digits(
            &digits,
            grouping,
            group_size,
            grouped_zero_width(spec, surrounding_bytes),
            surrounding_bytes,
        )?;
    }
    Ok(format!(
        "{}{prefix}{digits}",
        sign_prefix(negative, spec.sign)
    ))
}

fn format_number(value: f64, spec: &FormatSpec) -> Result<String, NativeTextFormatError> {
    if matches!(
        spec.kind,
        None | Some('f' | 'F' | 'e' | 'E' | '%' | 'g' | 'n' | 'G')
    ) && spec
        .precision
        .is_some_and(|precision| precision > NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES)
    {
        return Err(NativeTextFormatError::ResultTooLarge {
            maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES,
        });
    }
    let mut negative = value.is_sign_negative();
    let precision = spec.precision.unwrap_or(6);
    let magnitude = value.abs();
    let mut formatted = match spec.kind {
        None if spec.precision.is_none() => python_float(magnitude),
        None => format_general(magnitude, precision, false, spec.alternate, true),
        Some('f' | 'F') => format_fixed(magnitude, precision, spec.alternate),
        Some('e') => format_exponent(magnitude, precision, false, spec.alternate),
        Some('E') => format_exponent(magnitude, precision, true, spec.alternate),
        Some('%') => format!(
            "{}%",
            format_fixed(magnitude * 100.0, precision, spec.alternate)
        ),
        Some('g' | 'n') => format_general(magnitude, precision, false, spec.alternate, false),
        Some('G') => format_general(magnitude, precision, true, spec.alternate, false),
        _ => {
            return Err(NativeTextFormatError::InvalidTemplate(
                "unsupported number format type".to_owned(),
            ));
        }
    };
    if spec.coerce_negative_zero && formatted_value_is_zero(&formatted) {
        negative = false;
    }
    if let Some(grouping) = spec.grouping {
        formatted = group_float_integer_part(
            &formatted,
            grouping,
            spec,
            sign_prefix(negative, spec.sign).len(),
        )?;
    }
    Ok(format!("{}{formatted}", sign_prefix(negative, spec.sign)))
}

fn format_fixed(value: f64, precision: usize, alternate: bool) -> String {
    let mut value = format!("{value:.precision$}");
    if alternate && precision == 0 && !value.contains('.') {
        value.push('.');
    }
    value
}

fn format_exponent(value: f64, precision: usize, uppercase: bool, alternate: bool) -> String {
    let mut value = format!("{value:.precision$e}");
    if alternate && precision == 0 {
        if let Some(exponent) = value.find('e') {
            value.insert(exponent, '.');
        }
    }
    normalize_exponent(value, uppercase)
}

fn format_general(
    value: f64,
    precision: usize,
    uppercase: bool,
    alternate: bool,
    no_type: bool,
) -> String {
    if !value.is_finite() {
        let value = python_float(value);
        return if uppercase {
            value.to_ascii_uppercase()
        } else {
            value
        };
    }
    let precision = precision.max(1);
    let exponent = rounded_decimal_exponent(value, precision);
    let positive_cutoff = i32::try_from(precision)
        .unwrap_or(i32::MAX)
        .saturating_sub(i32::from(no_type));
    if exponent < -4 || exponent >= positive_cutoff {
        let mut value = format_exponent(value, precision.saturating_sub(1), uppercase, alternate);
        if !alternate {
            trim_fraction_zeros(&mut value);
        }
        value
    } else {
        let fractional_digits =
            usize::try_from((i32::try_from(precision).unwrap_or(i32::MAX) - exponent - 1).max(0))
                .unwrap_or(0);
        let mut value = format_fixed(value, fractional_digits, alternate);
        if !alternate {
            trim_fraction_zeros(&mut value);
        }
        if no_type && !value.contains('.') {
            value.push_str(".0");
        }
        value
    }
}

fn rounded_decimal_exponent(value: f64, precision: usize) -> i32 {
    let scientific = format!(
        "{value:.precision$e}",
        precision = precision.saturating_sub(1)
    );
    scientific
        .split_once('e')
        .and_then(|(_, exponent)| exponent.parse::<i32>().ok())
        .unwrap_or_else(|| value.log10().floor() as i32)
}

fn normalize_exponent(value: String, uppercase: bool) -> String {
    let marker = value.find('e').or_else(|| value.find('E'));
    let Some(marker) = marker else {
        return if uppercase {
            value.to_ascii_uppercase()
        } else {
            value
        };
    };
    let mantissa = value.get(..marker).unwrap_or_default();
    let exponent = value.get(marker + 1..).unwrap_or_default();
    let parsed = exponent.parse::<i32>().unwrap_or_default();
    let sign = if parsed < 0 { '-' } else { '+' };
    let magnitude = parsed.unsigned_abs();
    let marker = if uppercase { 'E' } else { 'e' };
    format!("{mantissa}{marker}{sign}{magnitude:02}")
}

fn trim_fraction_zeros(value: &mut String) {
    let exponent = value.find(['e', 'E']).unwrap_or(value.len());
    let Some(decimal) = value.get(..exponent).and_then(|prefix| prefix.find('.')) else {
        return;
    };
    let mut end = exponent;
    while end > decimal + 1 && value.as_bytes().get(end - 1) == Some(&b'0') {
        end -= 1;
    }
    if end == decimal + 1 {
        end = decimal;
    }
    value.replace_range(end..exponent, "");
}

fn grouped_zero_width(spec: &FormatSpec, surrounding_bytes: usize) -> usize {
    let align = spec.align.unwrap_or(if spec.zero { '=' } else { '>' });
    if spec.fill == '0' && align == '=' {
        spec.width.unwrap_or(0).saturating_sub(surrounding_bytes)
    } else {
        0
    }
}

fn grouped_digit_count(
    digit_count: usize,
    group_size: usize,
) -> Result<usize, NativeTextFormatError> {
    digit_count
        .checked_add(digit_count.saturating_sub(1) / group_size)
        .ok_or(NativeTextFormatError::ResultTooLarge {
            maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES,
        })
}

fn group_digits(
    value: &str,
    separator: char,
    group_size: usize,
    minimum_width: usize,
    surrounding_bytes: usize,
) -> Result<String, NativeTextFormatError> {
    let character_count = value.chars().count();
    if minimum_width > NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES {
        return Err(NativeTextFormatError::ResultTooLarge {
            maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES,
        });
    }
    let mut lower = character_count;
    let mut upper = character_count.max(minimum_width);
    while lower < upper {
        let middle = lower + (upper - lower) / 2;
        if grouped_digit_count(middle, group_size)? < minimum_width {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    let digit_count = lower;
    let zero_count = digit_count - character_count;
    let separator_count = digit_count.saturating_sub(1) / group_size;
    let output_bytes = separator_count
        .checked_mul(separator.len_utf8())
        .and_then(|bytes| bytes.checked_add(zero_count))
        .and_then(|bytes| bytes.checked_add(value.len()))
        .ok_or(NativeTextFormatError::ResultTooLarge {
            maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES,
        })?;
    if output_bytes
        .checked_add(surrounding_bytes)
        .is_none_or(|bytes| bytes > NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES)
    {
        return Err(NativeTextFormatError::ResultTooLarge {
            maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES,
        });
    }
    // A separator can make the first fitting grouped result wider than the requested width.
    let first_group = digit_count % group_size;
    let mut output = String::with_capacity(output_bytes);
    let characters = std::iter::repeat_n('0', zero_count).chain(value.chars());
    for (index, character) in characters.enumerate() {
        if index != 0
            && (index == first_group
                || (index > first_group && (index - first_group).is_multiple_of(group_size)))
        {
            output.push(separator);
        }
        output.push(character);
    }
    Ok(output)
}

fn group_float_integer_part(
    value: &str,
    separator: char,
    spec: &FormatSpec,
    sign_bytes: usize,
) -> Result<String, NativeTextFormatError> {
    let integer_end = value.find(['.', 'e', 'E', '%']).unwrap_or(value.len());
    let integer = value.get(..integer_end).unwrap_or_default();
    let suffix = value.get(integer_end..).unwrap_or_default();
    let surrounding_bytes = sign_bytes + suffix.len();
    let minimum_width = if integer.bytes().all(|byte| byte.is_ascii_digit()) {
        grouped_zero_width(spec, surrounding_bytes)
    } else {
        0
    };
    let integer = group_digits(integer, separator, 3, minimum_width, surrounding_bytes)?;
    Ok(format!("{integer}{suffix}"))
}

fn formatted_value_is_zero(value: &str) -> bool {
    let value = value.strip_suffix('%').unwrap_or(value);
    value.parse::<f64>().is_ok_and(|value| value == 0.0)
}

fn pad_numeric_after_prefix(
    value: String,
    fill: char,
    padding: usize,
) -> Result<String, NativeTextFormatError> {
    let sign_length = usize::from(
        value
            .chars()
            .next()
            .is_some_and(|value| matches!(value, '+' | '-' | ' ')),
    );
    let prefix_length = if value.get(sign_length..).is_some_and(|value| {
        value.starts_with("0x")
            || value.starts_with("0X")
            || value.starts_with("0o")
            || value.starts_with("0b")
    }) {
        2
    } else {
        0
    };
    let split = sign_length + prefix_length;
    let (prefix, digits) = value.split_at(split);
    let repeated = repeat_fill(fill, padding, value.len())?;
    Ok(format!("{prefix}{repeated}{digits}"))
}

fn pad_value(
    value: String,
    fill: char,
    left: usize,
    right: usize,
) -> Result<String, NativeTextFormatError> {
    let left = repeat_fill(fill, left, value.len())?;
    let size =
        left.len()
            .checked_add(value.len())
            .ok_or(NativeTextFormatError::ResultTooLarge {
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES,
            })?;
    let right = repeat_fill(fill, right, size)?;
    Ok(format!("{left}{value}{right}"))
}

fn repeat_fill(
    fill: char,
    count: usize,
    existing_bytes: usize,
) -> Result<String, NativeTextFormatError> {
    let repeated_bytes = fill
        .len_utf8()
        .checked_mul(count)
        .and_then(|bytes| bytes.checked_add(existing_bytes))
        .ok_or(NativeTextFormatError::ResultTooLarge {
            maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES,
        })?;
    if repeated_bytes > NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES {
        return Err(NativeTextFormatError::ResultTooLarge {
            maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES,
        });
    }
    Ok(fill.to_string().repeat(count))
}

fn sign_prefix(negative: bool, sign: Option<char>) -> &'static str {
    if negative {
        "-"
    } else {
        match sign {
            Some('+') => "+",
            Some(' ') => " ",
            _ => "",
        }
    }
}

fn append_bounded(output: &mut String, value: &str) -> Result<(), NativeTextFormatError> {
    let size =
        output
            .len()
            .checked_add(value.len())
            .ok_or(NativeTextFormatError::ResultTooLarge {
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES,
            })?;
    if size > NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES {
        return Err(NativeTextFormatError::ResultTooLarge {
            maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES,
        });
    }
    output.push_str(value);
    Ok(())
}

fn check_cancellation(cancellation: &CancellationToken) -> Result<(), NativeTextFormatError> {
    cancellation
        .check()
        .map_err(|_| NativeTextFormatError::Cancelled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NativeTextRegex, NativeTextRegexFlags};
    use serde::Serialize;
    use sha2::{Digest, Sha256};
    use std::error::Error;
    use std::fs;
    use std::path::Path;

    fn primitive(value: NativePrimitive) -> NativeValue {
        NativeValue::Primitive { value }
    }

    #[test]
    fn python_312_numeric_zero_padding_interleaves_decimal_and_radix_groups()
    -> Result<(), Box<dyn Error>> {
        for (template, value, expected) in [
            ("{a:04,d}", NativePrimitive::Integer(1), "0,001"),
            ("{a:05,d}", NativePrimitive::Integer(1), "0,001"),
            ("{a:06,d}", NativePrimitive::Integer(1), "00,001"),
            ("{a:07,d}", NativePrimitive::Integer(1), "000,001"),
            ("{a:08,d}", NativePrimitive::Integer(1), "0,000,001"),
            ("{a:09,d}", NativePrimitive::Integer(1), "0,000,001"),
            ("{a:010,d}", NativePrimitive::Integer(1), "00,000,001"),
            ("{a:014,d}", NativePrimitive::Integer(1), "00,000,000,001"),
            ("{a:04_d}", NativePrimitive::Integer(1), "0_001"),
            ("{a:05_d}", NativePrimitive::Integer(1), "0_001"),
            ("{a:06_d}", NativePrimitive::Integer(1), "00_001"),
            ("{a:07_d}", NativePrimitive::Integer(1), "000_001"),
            ("{a:08_d}", NativePrimitive::Integer(1), "0_000_001"),
            ("{a:09_d}", NativePrimitive::Integer(1), "0_000_001"),
            ("{a:010_d}", NativePrimitive::Integer(1), "00_000_001"),
            ("{a:014_d}", NativePrimitive::Integer(1), "00_000_000_001"),
            ("{a:04,d}", NativePrimitive::Integer(1000), "1,000"),
            ("{a:05,d}", NativePrimitive::Integer(1000), "1,000"),
            ("{a:06,d}", NativePrimitive::Integer(1000), "01,000"),
            ("{a:07,d}", NativePrimitive::Integer(1000), "001,000"),
            ("{a:08,d}", NativePrimitive::Integer(1000), "0,001,000"),
            ("{a:09,d}", NativePrimitive::Integer(1000), "0,001,000"),
            ("{a:010,d}", NativePrimitive::Integer(1000), "00,001,000"),
            (
                "{a:014,d}",
                NativePrimitive::Integer(1000),
                "00,000,001,000",
            ),
            ("{a:04_d}", NativePrimitive::Integer(1000), "1_000"),
            ("{a:05_d}", NativePrimitive::Integer(1000), "1_000"),
            ("{a:06_d}", NativePrimitive::Integer(1000), "01_000"),
            ("{a:07_d}", NativePrimitive::Integer(1000), "001_000"),
            ("{a:08_d}", NativePrimitive::Integer(1000), "0_001_000"),
            ("{a:09_d}", NativePrimitive::Integer(1000), "0_001_000"),
            ("{a:010_d}", NativePrimitive::Integer(1000), "00_001_000"),
            (
                "{a:014_d}",
                NativePrimitive::Integer(1000),
                "00_000_001_000",
            ),
            ("{a:04,d}", NativePrimitive::Integer(12345), "12,345"),
            ("{a:05,d}", NativePrimitive::Integer(12345), "12,345"),
            ("{a:06,d}", NativePrimitive::Integer(12345), "12,345"),
            ("{a:07,d}", NativePrimitive::Integer(12345), "012,345"),
            ("{a:08,d}", NativePrimitive::Integer(12345), "0,012,345"),
            ("{a:09,d}", NativePrimitive::Integer(12345), "0,012,345"),
            ("{a:010,d}", NativePrimitive::Integer(12345), "00,012,345"),
            (
                "{a:014,d}",
                NativePrimitive::Integer(12345),
                "00,000,012,345",
            ),
            ("{a:04_d}", NativePrimitive::Integer(12345), "12_345"),
            ("{a:05_d}", NativePrimitive::Integer(12345), "12_345"),
            ("{a:06_d}", NativePrimitive::Integer(12345), "12_345"),
            ("{a:07_d}", NativePrimitive::Integer(12345), "012_345"),
            ("{a:08_d}", NativePrimitive::Integer(12345), "0_012_345"),
            ("{a:09_d}", NativePrimitive::Integer(12345), "0_012_345"),
            ("{a:010_d}", NativePrimitive::Integer(12345), "00_012_345"),
            (
                "{a:014_d}",
                NativePrimitive::Integer(12345),
                "00_000_012_345",
            ),
            ("{a:04,d}", NativePrimitive::Integer(-12345), "-12,345"),
            ("{a:05,d}", NativePrimitive::Integer(-12345), "-12,345"),
            ("{a:06,d}", NativePrimitive::Integer(-12345), "-12,345"),
            ("{a:07,d}", NativePrimitive::Integer(-12345), "-12,345"),
            ("{a:08,d}", NativePrimitive::Integer(-12345), "-012,345"),
            ("{a:09,d}", NativePrimitive::Integer(-12345), "-0,012,345"),
            ("{a:010,d}", NativePrimitive::Integer(-12345), "-0,012,345"),
            (
                "{a:014,d}",
                NativePrimitive::Integer(-12345),
                "-0,000,012,345",
            ),
            ("{a:04_d}", NativePrimitive::Integer(-12345), "-12_345"),
            ("{a:05_d}", NativePrimitive::Integer(-12345), "-12_345"),
            ("{a:06_d}", NativePrimitive::Integer(-12345), "-12_345"),
            ("{a:07_d}", NativePrimitive::Integer(-12345), "-12_345"),
            ("{a:08_d}", NativePrimitive::Integer(-12345), "-012_345"),
            ("{a:09_d}", NativePrimitive::Integer(-12345), "-0_012_345"),
            ("{a:010_d}", NativePrimitive::Integer(-12345), "-0_012_345"),
            (
                "{a:014_d}",
                NativePrimitive::Integer(-12345),
                "-0_000_012_345",
            ),
            ("{a:04,d}", NativePrimitive::Boolean(true), "0,001"),
            ("{a:05,d}", NativePrimitive::Boolean(true), "0,001"),
            ("{a:06,d}", NativePrimitive::Boolean(true), "00,001"),
            ("{a:07,d}", NativePrimitive::Boolean(true), "000,001"),
            ("{a:08,d}", NativePrimitive::Boolean(true), "0,000,001"),
            ("{a:09,d}", NativePrimitive::Boolean(true), "0,000,001"),
            ("{a:010,d}", NativePrimitive::Boolean(true), "00,000,001"),
            (
                "{a:014,d}",
                NativePrimitive::Boolean(true),
                "00,000,000,001",
            ),
            ("{a:04_d}", NativePrimitive::Boolean(true), "0_001"),
            ("{a:05_d}", NativePrimitive::Boolean(true), "0_001"),
            ("{a:06_d}", NativePrimitive::Boolean(true), "00_001"),
            ("{a:07_d}", NativePrimitive::Boolean(true), "000_001"),
            ("{a:08_d}", NativePrimitive::Boolean(true), "0_000_001"),
            ("{a:09_d}", NativePrimitive::Boolean(true), "0_000_001"),
            ("{a:010_d}", NativePrimitive::Boolean(true), "00_000_001"),
            (
                "{a:014_d}",
                NativePrimitive::Boolean(true),
                "00_000_000_001",
            ),
            (
                "{a:04,d}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9,223,372,036,854,775,808",
            ),
            (
                "{a:05,d}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9,223,372,036,854,775,808",
            ),
            (
                "{a:06,d}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9,223,372,036,854,775,808",
            ),
            (
                "{a:07,d}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9,223,372,036,854,775,808",
            ),
            (
                "{a:08,d}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9,223,372,036,854,775,808",
            ),
            (
                "{a:09,d}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9,223,372,036,854,775,808",
            ),
            (
                "{a:010,d}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9,223,372,036,854,775,808",
            ),
            (
                "{a:014,d}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9,223,372,036,854,775,808",
            ),
            (
                "{a:04_d}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9_223_372_036_854_775_808",
            ),
            (
                "{a:05_d}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9_223_372_036_854_775_808",
            ),
            (
                "{a:06_d}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9_223_372_036_854_775_808",
            ),
            (
                "{a:07_d}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9_223_372_036_854_775_808",
            ),
            (
                "{a:08_d}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9_223_372_036_854_775_808",
            ),
            (
                "{a:09_d}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9_223_372_036_854_775_808",
            ),
            (
                "{a:010_d}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9_223_372_036_854_775_808",
            ),
            (
                "{a:014_d}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9_223_372_036_854_775_808",
            ),
            ("{a:08_b}", NativePrimitive::Integer(7), "000_0111"),
            ("{a:010_b}", NativePrimitive::Integer(7), "0_0000_0111"),
            ("{a:012_b}", NativePrimitive::Integer(7), "00_0000_0111"),
            ("{a:+#08_b}", NativePrimitive::Integer(7), "+0b0_0111"),
            ("{a:+#010_b}", NativePrimitive::Integer(7), "+0b00_0111"),
            ("{a:+#012_b}", NativePrimitive::Integer(7), "+0b0000_0111"),
            ("{a:0=+#08_b}", NativePrimitive::Integer(7), "+0b0_0111"),
            ("{a:0=+#010_b}", NativePrimitive::Integer(7), "+0b00_0111"),
            ("{a:0=+#012_b}", NativePrimitive::Integer(7), "+0b0000_0111"),
            ("{a:08_o}", NativePrimitive::Integer(7), "000_0007"),
            ("{a:010_o}", NativePrimitive::Integer(7), "0_0000_0007"),
            ("{a:012_o}", NativePrimitive::Integer(7), "00_0000_0007"),
            ("{a:+#08_o}", NativePrimitive::Integer(7), "+0o0_0007"),
            ("{a:+#010_o}", NativePrimitive::Integer(7), "+0o00_0007"),
            ("{a:+#012_o}", NativePrimitive::Integer(7), "+0o0000_0007"),
            ("{a:0=+#08_o}", NativePrimitive::Integer(7), "+0o0_0007"),
            ("{a:0=+#010_o}", NativePrimitive::Integer(7), "+0o00_0007"),
            ("{a:0=+#012_o}", NativePrimitive::Integer(7), "+0o0000_0007"),
            ("{a:08_x}", NativePrimitive::Integer(7), "000_0007"),
            ("{a:010_x}", NativePrimitive::Integer(7), "0_0000_0007"),
            ("{a:012_x}", NativePrimitive::Integer(7), "00_0000_0007"),
            ("{a:+#08_x}", NativePrimitive::Integer(7), "+0x0_0007"),
            ("{a:+#010_x}", NativePrimitive::Integer(7), "+0x00_0007"),
            ("{a:+#012_x}", NativePrimitive::Integer(7), "+0x0000_0007"),
            ("{a:0=+#08_x}", NativePrimitive::Integer(7), "+0x0_0007"),
            ("{a:0=+#010_x}", NativePrimitive::Integer(7), "+0x00_0007"),
            ("{a:0=+#012_x}", NativePrimitive::Integer(7), "+0x0000_0007"),
            ("{a:08_X}", NativePrimitive::Integer(7), "000_0007"),
            ("{a:010_X}", NativePrimitive::Integer(7), "0_0000_0007"),
            ("{a:012_X}", NativePrimitive::Integer(7), "00_0000_0007"),
            ("{a:+#08_X}", NativePrimitive::Integer(7), "+0X0_0007"),
            ("{a:+#010_X}", NativePrimitive::Integer(7), "+0X00_0007"),
            ("{a:+#012_X}", NativePrimitive::Integer(7), "+0X0000_0007"),
            ("{a:0=+#08_X}", NativePrimitive::Integer(7), "+0X0_0007"),
            ("{a:0=+#010_X}", NativePrimitive::Integer(7), "+0X00_0007"),
            ("{a:0=+#012_X}", NativePrimitive::Integer(7), "+0X0000_0007"),
            ("{a:08_b}", NativePrimitive::Integer(-7), "-00_0111"),
            ("{a:010_b}", NativePrimitive::Integer(-7), "-0000_0111"),
            ("{a:012_b}", NativePrimitive::Integer(-7), "-0_0000_0111"),
            ("{a:+#08_b}", NativePrimitive::Integer(-7), "-0b0_0111"),
            ("{a:+#010_b}", NativePrimitive::Integer(-7), "-0b00_0111"),
            ("{a:+#012_b}", NativePrimitive::Integer(-7), "-0b0000_0111"),
            ("{a:0=+#08_b}", NativePrimitive::Integer(-7), "-0b0_0111"),
            ("{a:0=+#010_b}", NativePrimitive::Integer(-7), "-0b00_0111"),
            (
                "{a:0=+#012_b}",
                NativePrimitive::Integer(-7),
                "-0b0000_0111",
            ),
            ("{a:08_o}", NativePrimitive::Integer(-7), "-00_0007"),
            ("{a:010_o}", NativePrimitive::Integer(-7), "-0000_0007"),
            ("{a:012_o}", NativePrimitive::Integer(-7), "-0_0000_0007"),
            ("{a:+#08_o}", NativePrimitive::Integer(-7), "-0o0_0007"),
            ("{a:+#010_o}", NativePrimitive::Integer(-7), "-0o00_0007"),
            ("{a:+#012_o}", NativePrimitive::Integer(-7), "-0o0000_0007"),
            ("{a:0=+#08_o}", NativePrimitive::Integer(-7), "-0o0_0007"),
            ("{a:0=+#010_o}", NativePrimitive::Integer(-7), "-0o00_0007"),
            (
                "{a:0=+#012_o}",
                NativePrimitive::Integer(-7),
                "-0o0000_0007",
            ),
            ("{a:08_x}", NativePrimitive::Integer(-7), "-00_0007"),
            ("{a:010_x}", NativePrimitive::Integer(-7), "-0000_0007"),
            ("{a:012_x}", NativePrimitive::Integer(-7), "-0_0000_0007"),
            ("{a:+#08_x}", NativePrimitive::Integer(-7), "-0x0_0007"),
            ("{a:+#010_x}", NativePrimitive::Integer(-7), "-0x00_0007"),
            ("{a:+#012_x}", NativePrimitive::Integer(-7), "-0x0000_0007"),
            ("{a:0=+#08_x}", NativePrimitive::Integer(-7), "-0x0_0007"),
            ("{a:0=+#010_x}", NativePrimitive::Integer(-7), "-0x00_0007"),
            (
                "{a:0=+#012_x}",
                NativePrimitive::Integer(-7),
                "-0x0000_0007",
            ),
            ("{a:08_X}", NativePrimitive::Integer(-7), "-00_0007"),
            ("{a:010_X}", NativePrimitive::Integer(-7), "-0000_0007"),
            ("{a:012_X}", NativePrimitive::Integer(-7), "-0_0000_0007"),
            ("{a:+#08_X}", NativePrimitive::Integer(-7), "-0X0_0007"),
            ("{a:+#010_X}", NativePrimitive::Integer(-7), "-0X00_0007"),
            ("{a:+#012_X}", NativePrimitive::Integer(-7), "-0X0000_0007"),
            ("{a:0=+#08_X}", NativePrimitive::Integer(-7), "-0X0_0007"),
            ("{a:0=+#010_X}", NativePrimitive::Integer(-7), "-0X00_0007"),
            (
                "{a:0=+#012_X}",
                NativePrimitive::Integer(-7),
                "-0X0000_0007",
            ),
            (
                "{a:08_b}",
                NativePrimitive::Integer(74565),
                "1_0010_0011_0100_0101",
            ),
            (
                "{a:010_b}",
                NativePrimitive::Integer(74565),
                "1_0010_0011_0100_0101",
            ),
            (
                "{a:012_b}",
                NativePrimitive::Integer(74565),
                "1_0010_0011_0100_0101",
            ),
            (
                "{a:+#08_b}",
                NativePrimitive::Integer(74565),
                "+0b1_0010_0011_0100_0101",
            ),
            (
                "{a:+#010_b}",
                NativePrimitive::Integer(74565),
                "+0b1_0010_0011_0100_0101",
            ),
            (
                "{a:+#012_b}",
                NativePrimitive::Integer(74565),
                "+0b1_0010_0011_0100_0101",
            ),
            (
                "{a:0=+#08_b}",
                NativePrimitive::Integer(74565),
                "+0b1_0010_0011_0100_0101",
            ),
            (
                "{a:0=+#010_b}",
                NativePrimitive::Integer(74565),
                "+0b1_0010_0011_0100_0101",
            ),
            (
                "{a:0=+#012_b}",
                NativePrimitive::Integer(74565),
                "+0b1_0010_0011_0100_0101",
            ),
            ("{a:08_o}", NativePrimitive::Integer(74565), "022_1505"),
            ("{a:010_o}", NativePrimitive::Integer(74565), "0_0022_1505"),
            ("{a:012_o}", NativePrimitive::Integer(74565), "00_0022_1505"),
            ("{a:+#08_o}", NativePrimitive::Integer(74565), "+0o22_1505"),
            ("{a:+#010_o}", NativePrimitive::Integer(74565), "+0o22_1505"),
            (
                "{a:+#012_o}",
                NativePrimitive::Integer(74565),
                "+0o0022_1505",
            ),
            (
                "{a:0=+#08_o}",
                NativePrimitive::Integer(74565),
                "+0o22_1505",
            ),
            (
                "{a:0=+#010_o}",
                NativePrimitive::Integer(74565),
                "+0o22_1505",
            ),
            (
                "{a:0=+#012_o}",
                NativePrimitive::Integer(74565),
                "+0o0022_1505",
            ),
            ("{a:08_x}", NativePrimitive::Integer(74565), "001_2345"),
            ("{a:010_x}", NativePrimitive::Integer(74565), "0_0001_2345"),
            ("{a:012_x}", NativePrimitive::Integer(74565), "00_0001_2345"),
            ("{a:+#08_x}", NativePrimitive::Integer(74565), "+0x1_2345"),
            ("{a:+#010_x}", NativePrimitive::Integer(74565), "+0x01_2345"),
            (
                "{a:+#012_x}",
                NativePrimitive::Integer(74565),
                "+0x0001_2345",
            ),
            ("{a:0=+#08_x}", NativePrimitive::Integer(74565), "+0x1_2345"),
            (
                "{a:0=+#010_x}",
                NativePrimitive::Integer(74565),
                "+0x01_2345",
            ),
            (
                "{a:0=+#012_x}",
                NativePrimitive::Integer(74565),
                "+0x0001_2345",
            ),
            ("{a:08_X}", NativePrimitive::Integer(74565), "001_2345"),
            ("{a:010_X}", NativePrimitive::Integer(74565), "0_0001_2345"),
            ("{a:012_X}", NativePrimitive::Integer(74565), "00_0001_2345"),
            ("{a:+#08_X}", NativePrimitive::Integer(74565), "+0X1_2345"),
            ("{a:+#010_X}", NativePrimitive::Integer(74565), "+0X01_2345"),
            (
                "{a:+#012_X}",
                NativePrimitive::Integer(74565),
                "+0X0001_2345",
            ),
            ("{a:0=+#08_X}", NativePrimitive::Integer(74565), "+0X1_2345"),
            (
                "{a:0=+#010_X}",
                NativePrimitive::Integer(74565),
                "+0X01_2345",
            ),
            (
                "{a:0=+#012_X}",
                NativePrimitive::Integer(74565),
                "+0X0001_2345",
            ),
        ] {
            let values = BTreeMap::from([("a".to_owned(), primitive(value))]);
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default())?,
                expected,
                "original grouped padding: {template}, {values:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn python_312_grouped_float_zero_padding_preserves_fraction_exponent_and_percent()
    -> Result<(), Box<dyn Error>> {
        for (template, value, expected) in [
            ("{a:08,.0f}", NativePrimitive::Number(0.0), "0,000,000"),
            ("{a:012,.0f}", NativePrimitive::Number(0.0), "0,000,000,000"),
            (
                "{a:014,.0f}",
                NativePrimitive::Number(0.0),
                "00,000,000,000",
            ),
            ("{a:08,.2f}", NativePrimitive::Number(0.0), "0,000.00"),
            ("{a:012,.2f}", NativePrimitive::Number(0.0), "0,000,000.00"),
            (
                "{a:014,.2f}",
                NativePrimitive::Number(0.0),
                "000,000,000.00",
            ),
            ("{a:08,.1e}", NativePrimitive::Number(0.0), "00.0e+00"),
            ("{a:012,.1e}", NativePrimitive::Number(0.0), "00,000.0e+00"),
            (
                "{a:014,.1e}",
                NativePrimitive::Number(0.0),
                "0,000,000.0e+00",
            ),
            ("{a:08,.3g}", NativePrimitive::Number(0.0), "0,000,000"),
            ("{a:012,.3g}", NativePrimitive::Number(0.0), "0,000,000,000"),
            (
                "{a:014,.3g}",
                NativePrimitive::Number(0.0),
                "00,000,000,000",
            ),
            ("{a:08,.0%}", NativePrimitive::Number(0.0), "000,000%"),
            ("{a:012,.0%}", NativePrimitive::Number(0.0), "000,000,000%"),
            (
                "{a:014,.0%}",
                NativePrimitive::Number(0.0),
                "0,000,000,000%",
            ),
            ("{a:08,.1%}", NativePrimitive::Number(0.0), "0,000.0%"),
            ("{a:012,.1%}", NativePrimitive::Number(0.0), "0,000,000.0%"),
            (
                "{a:014,.1%}",
                NativePrimitive::Number(0.0),
                "000,000,000.0%",
            ),
            ("{a:08_.0f}", NativePrimitive::Number(0.0), "0_000_000"),
            ("{a:012_.0f}", NativePrimitive::Number(0.0), "0_000_000_000"),
            (
                "{a:014_.0f}",
                NativePrimitive::Number(0.0),
                "00_000_000_000",
            ),
            ("{a:08_.2f}", NativePrimitive::Number(0.0), "0_000.00"),
            ("{a:012_.2f}", NativePrimitive::Number(0.0), "0_000_000.00"),
            (
                "{a:014_.2f}",
                NativePrimitive::Number(0.0),
                "000_000_000.00",
            ),
            ("{a:08_.1e}", NativePrimitive::Number(0.0), "00.0e+00"),
            ("{a:012_.1e}", NativePrimitive::Number(0.0), "00_000.0e+00"),
            (
                "{a:014_.1e}",
                NativePrimitive::Number(0.0),
                "0_000_000.0e+00",
            ),
            ("{a:08_.3g}", NativePrimitive::Number(0.0), "0_000_000"),
            ("{a:012_.3g}", NativePrimitive::Number(0.0), "0_000_000_000"),
            (
                "{a:014_.3g}",
                NativePrimitive::Number(0.0),
                "00_000_000_000",
            ),
            ("{a:08_.0%}", NativePrimitive::Number(0.0), "000_000%"),
            ("{a:012_.0%}", NativePrimitive::Number(0.0), "000_000_000%"),
            (
                "{a:014_.0%}",
                NativePrimitive::Number(0.0),
                "0_000_000_000%",
            ),
            ("{a:08_.1%}", NativePrimitive::Number(0.0), "0_000.0%"),
            ("{a:012_.1%}", NativePrimitive::Number(0.0), "0_000_000.0%"),
            (
                "{a:014_.1%}",
                NativePrimitive::Number(0.0),
                "000_000_000.0%",
            ),
            ("{a:08,.0f}", NativePrimitive::Number(-0.0), "-000,000"),
            ("{a:012,.0f}", NativePrimitive::Number(-0.0), "-000,000,000"),
            (
                "{a:014,.0f}",
                NativePrimitive::Number(-0.0),
                "-0,000,000,000",
            ),
            ("{a:08,.2f}", NativePrimitive::Number(-0.0), "-0,000.00"),
            (
                "{a:012,.2f}",
                NativePrimitive::Number(-0.0),
                "-0,000,000.00",
            ),
            (
                "{a:014,.2f}",
                NativePrimitive::Number(-0.0),
                "-00,000,000.00",
            ),
            ("{a:08,.1e}", NativePrimitive::Number(-0.0), "-0.0e+00"),
            ("{a:012,.1e}", NativePrimitive::Number(-0.0), "-0,000.0e+00"),
            (
                "{a:014,.1e}",
                NativePrimitive::Number(-0.0),
                "-000,000.0e+00",
            ),
            ("{a:08,.3g}", NativePrimitive::Number(-0.0), "-000,000"),
            ("{a:012,.3g}", NativePrimitive::Number(-0.0), "-000,000,000"),
            (
                "{a:014,.3g}",
                NativePrimitive::Number(-0.0),
                "-0,000,000,000",
            ),
            ("{a:08,.0%}", NativePrimitive::Number(-0.0), "-00,000%"),
            ("{a:012,.0%}", NativePrimitive::Number(-0.0), "-00,000,000%"),
            (
                "{a:014,.0%}",
                NativePrimitive::Number(-0.0),
                "-0,000,000,000%",
            ),
            ("{a:08,.1%}", NativePrimitive::Number(-0.0), "-0,000.0%"),
            (
                "{a:012,.1%}",
                NativePrimitive::Number(-0.0),
                "-0,000,000.0%",
            ),
            (
                "{a:014,.1%}",
                NativePrimitive::Number(-0.0),
                "-00,000,000.0%",
            ),
            ("{a:08_.0f}", NativePrimitive::Number(-0.0), "-000_000"),
            ("{a:012_.0f}", NativePrimitive::Number(-0.0), "-000_000_000"),
            (
                "{a:014_.0f}",
                NativePrimitive::Number(-0.0),
                "-0_000_000_000",
            ),
            ("{a:08_.2f}", NativePrimitive::Number(-0.0), "-0_000.00"),
            (
                "{a:012_.2f}",
                NativePrimitive::Number(-0.0),
                "-0_000_000.00",
            ),
            (
                "{a:014_.2f}",
                NativePrimitive::Number(-0.0),
                "-00_000_000.00",
            ),
            ("{a:08_.1e}", NativePrimitive::Number(-0.0), "-0.0e+00"),
            ("{a:012_.1e}", NativePrimitive::Number(-0.0), "-0_000.0e+00"),
            (
                "{a:014_.1e}",
                NativePrimitive::Number(-0.0),
                "-000_000.0e+00",
            ),
            ("{a:08_.3g}", NativePrimitive::Number(-0.0), "-000_000"),
            ("{a:012_.3g}", NativePrimitive::Number(-0.0), "-000_000_000"),
            (
                "{a:014_.3g}",
                NativePrimitive::Number(-0.0),
                "-0_000_000_000",
            ),
            ("{a:08_.0%}", NativePrimitive::Number(-0.0), "-00_000%"),
            ("{a:012_.0%}", NativePrimitive::Number(-0.0), "-00_000_000%"),
            (
                "{a:014_.0%}",
                NativePrimitive::Number(-0.0),
                "-0_000_000_000%",
            ),
            ("{a:08_.1%}", NativePrimitive::Number(-0.0), "-0_000.0%"),
            (
                "{a:012_.1%}",
                NativePrimitive::Number(-0.0),
                "-0_000_000.0%",
            ),
            (
                "{a:014_.1%}",
                NativePrimitive::Number(-0.0),
                "-00_000_000.0%",
            ),
            ("{a:08,.0f}", NativePrimitive::Number(-1.25), "-000,001"),
            (
                "{a:012,.0f}",
                NativePrimitive::Number(-1.25),
                "-000,000,001",
            ),
            (
                "{a:014,.0f}",
                NativePrimitive::Number(-1.25),
                "-0,000,000,001",
            ),
            ("{a:08,.2f}", NativePrimitive::Number(-1.25), "-0,001.25"),
            (
                "{a:012,.2f}",
                NativePrimitive::Number(-1.25),
                "-0,000,001.25",
            ),
            (
                "{a:014,.2f}",
                NativePrimitive::Number(-1.25),
                "-00,000,001.25",
            ),
            ("{a:08,.1e}", NativePrimitive::Number(-1.25), "-1.2e+00"),
            (
                "{a:012,.1e}",
                NativePrimitive::Number(-1.25),
                "-0,001.2e+00",
            ),
            (
                "{a:014,.1e}",
                NativePrimitive::Number(-1.25),
                "-000,001.2e+00",
            ),
            ("{a:08,.3g}", NativePrimitive::Number(-1.25), "-0,001.25"),
            (
                "{a:012,.3g}",
                NativePrimitive::Number(-1.25),
                "-0,000,001.25",
            ),
            (
                "{a:014,.3g}",
                NativePrimitive::Number(-1.25),
                "-00,000,001.25",
            ),
            ("{a:08,.0%}", NativePrimitive::Number(-1.25), "-00,125%"),
            (
                "{a:012,.0%}",
                NativePrimitive::Number(-1.25),
                "-00,000,125%",
            ),
            (
                "{a:014,.0%}",
                NativePrimitive::Number(-1.25),
                "-0,000,000,125%",
            ),
            ("{a:08,.1%}", NativePrimitive::Number(-1.25), "-0,125.0%"),
            (
                "{a:012,.1%}",
                NativePrimitive::Number(-1.25),
                "-0,000,125.0%",
            ),
            (
                "{a:014,.1%}",
                NativePrimitive::Number(-1.25),
                "-00,000,125.0%",
            ),
            ("{a:08_.0f}", NativePrimitive::Number(-1.25), "-000_001"),
            (
                "{a:012_.0f}",
                NativePrimitive::Number(-1.25),
                "-000_000_001",
            ),
            (
                "{a:014_.0f}",
                NativePrimitive::Number(-1.25),
                "-0_000_000_001",
            ),
            ("{a:08_.2f}", NativePrimitive::Number(-1.25), "-0_001.25"),
            (
                "{a:012_.2f}",
                NativePrimitive::Number(-1.25),
                "-0_000_001.25",
            ),
            (
                "{a:014_.2f}",
                NativePrimitive::Number(-1.25),
                "-00_000_001.25",
            ),
            ("{a:08_.1e}", NativePrimitive::Number(-1.25), "-1.2e+00"),
            (
                "{a:012_.1e}",
                NativePrimitive::Number(-1.25),
                "-0_001.2e+00",
            ),
            (
                "{a:014_.1e}",
                NativePrimitive::Number(-1.25),
                "-000_001.2e+00",
            ),
            ("{a:08_.3g}", NativePrimitive::Number(-1.25), "-0_001.25"),
            (
                "{a:012_.3g}",
                NativePrimitive::Number(-1.25),
                "-0_000_001.25",
            ),
            (
                "{a:014_.3g}",
                NativePrimitive::Number(-1.25),
                "-00_000_001.25",
            ),
            ("{a:08_.0%}", NativePrimitive::Number(-1.25), "-00_125%"),
            (
                "{a:012_.0%}",
                NativePrimitive::Number(-1.25),
                "-00_000_125%",
            ),
            (
                "{a:014_.0%}",
                NativePrimitive::Number(-1.25),
                "-0_000_000_125%",
            ),
            ("{a:08_.1%}", NativePrimitive::Number(-1.25), "-0_125.0%"),
            (
                "{a:012_.1%}",
                NativePrimitive::Number(-1.25),
                "-0_000_125.0%",
            ),
            (
                "{a:014_.1%}",
                NativePrimitive::Number(-1.25),
                "-00_000_125.0%",
            ),
            ("{a:08,.0f}", NativePrimitive::Number(12345.5), "0,012,346"),
            (
                "{a:012,.0f}",
                NativePrimitive::Number(12345.5),
                "0,000,012,346",
            ),
            (
                "{a:014,.0f}",
                NativePrimitive::Number(12345.5),
                "00,000,012,346",
            ),
            ("{a:08,.2f}", NativePrimitive::Number(12345.5), "12,345.50"),
            (
                "{a:012,.2f}",
                NativePrimitive::Number(12345.5),
                "0,012,345.50",
            ),
            (
                "{a:014,.2f}",
                NativePrimitive::Number(12345.5),
                "000,012,345.50",
            ),
            ("{a:08,.1e}", NativePrimitive::Number(12345.5), "01.2e+04"),
            (
                "{a:012,.1e}",
                NativePrimitive::Number(12345.5),
                "00,001.2e+04",
            ),
            (
                "{a:014,.1e}",
                NativePrimitive::Number(12345.5),
                "0,000,001.2e+04",
            ),
            ("{a:08,.3g}", NativePrimitive::Number(12345.5), "1.23e+04"),
            (
                "{a:012,.3g}",
                NativePrimitive::Number(12345.5),
                "0,001.23e+04",
            ),
            (
                "{a:014,.3g}",
                NativePrimitive::Number(12345.5),
                "000,001.23e+04",
            ),
            ("{a:08,.0%}", NativePrimitive::Number(12345.5), "1,234,550%"),
            (
                "{a:012,.0%}",
                NativePrimitive::Number(12345.5),
                "001,234,550%",
            ),
            (
                "{a:014,.0%}",
                NativePrimitive::Number(12345.5),
                "0,001,234,550%",
            ),
            (
                "{a:08,.1%}",
                NativePrimitive::Number(12345.5),
                "1,234,550.0%",
            ),
            (
                "{a:012,.1%}",
                NativePrimitive::Number(12345.5),
                "1,234,550.0%",
            ),
            (
                "{a:014,.1%}",
                NativePrimitive::Number(12345.5),
                "001,234,550.0%",
            ),
            ("{a:08_.0f}", NativePrimitive::Number(12345.5), "0_012_346"),
            (
                "{a:012_.0f}",
                NativePrimitive::Number(12345.5),
                "0_000_012_346",
            ),
            (
                "{a:014_.0f}",
                NativePrimitive::Number(12345.5),
                "00_000_012_346",
            ),
            ("{a:08_.2f}", NativePrimitive::Number(12345.5), "12_345.50"),
            (
                "{a:012_.2f}",
                NativePrimitive::Number(12345.5),
                "0_012_345.50",
            ),
            (
                "{a:014_.2f}",
                NativePrimitive::Number(12345.5),
                "000_012_345.50",
            ),
            ("{a:08_.1e}", NativePrimitive::Number(12345.5), "01.2e+04"),
            (
                "{a:012_.1e}",
                NativePrimitive::Number(12345.5),
                "00_001.2e+04",
            ),
            (
                "{a:014_.1e}",
                NativePrimitive::Number(12345.5),
                "0_000_001.2e+04",
            ),
            ("{a:08_.3g}", NativePrimitive::Number(12345.5), "1.23e+04"),
            (
                "{a:012_.3g}",
                NativePrimitive::Number(12345.5),
                "0_001.23e+04",
            ),
            (
                "{a:014_.3g}",
                NativePrimitive::Number(12345.5),
                "000_001.23e+04",
            ),
            ("{a:08_.0%}", NativePrimitive::Number(12345.5), "1_234_550%"),
            (
                "{a:012_.0%}",
                NativePrimitive::Number(12345.5),
                "001_234_550%",
            ),
            (
                "{a:014_.0%}",
                NativePrimitive::Number(12345.5),
                "0_001_234_550%",
            ),
            (
                "{a:08_.1%}",
                NativePrimitive::Number(12345.5),
                "1_234_550.0%",
            ),
            (
                "{a:012_.1%}",
                NativePrimitive::Number(12345.5),
                "1_234_550.0%",
            ),
            (
                "{a:014_.1%}",
                NativePrimitive::Number(12345.5),
                "001_234_550.0%",
            ),
            (
                "{a:08,.0f}",
                NativePrimitive::Number(1e+20),
                "100,000,000,000,000,000,000",
            ),
            (
                "{a:012,.0f}",
                NativePrimitive::Number(1e+20),
                "100,000,000,000,000,000,000",
            ),
            (
                "{a:014,.0f}",
                NativePrimitive::Number(1e+20),
                "100,000,000,000,000,000,000",
            ),
            (
                "{a:08,.2f}",
                NativePrimitive::Number(1e+20),
                "100,000,000,000,000,000,000.00",
            ),
            (
                "{a:012,.2f}",
                NativePrimitive::Number(1e+20),
                "100,000,000,000,000,000,000.00",
            ),
            (
                "{a:014,.2f}",
                NativePrimitive::Number(1e+20),
                "100,000,000,000,000,000,000.00",
            ),
            ("{a:08,.1e}", NativePrimitive::Number(1e+20), "01.0e+20"),
            (
                "{a:012,.1e}",
                NativePrimitive::Number(1e+20),
                "00,001.0e+20",
            ),
            (
                "{a:014,.1e}",
                NativePrimitive::Number(1e+20),
                "0,000,001.0e+20",
            ),
            ("{a:08,.3g}", NativePrimitive::Number(1e+20), "0,001e+20"),
            (
                "{a:012,.3g}",
                NativePrimitive::Number(1e+20),
                "0,000,001e+20",
            ),
            (
                "{a:014,.3g}",
                NativePrimitive::Number(1e+20),
                "00,000,001e+20",
            ),
            (
                "{a:08,.0%}",
                NativePrimitive::Number(1e+20),
                "10,000,000,000,000,000,000,000%",
            ),
            (
                "{a:012,.0%}",
                NativePrimitive::Number(1e+20),
                "10,000,000,000,000,000,000,000%",
            ),
            (
                "{a:014,.0%}",
                NativePrimitive::Number(1e+20),
                "10,000,000,000,000,000,000,000%",
            ),
            (
                "{a:08,.1%}",
                NativePrimitive::Number(1e+20),
                "10,000,000,000,000,000,000,000.0%",
            ),
            (
                "{a:012,.1%}",
                NativePrimitive::Number(1e+20),
                "10,000,000,000,000,000,000,000.0%",
            ),
            (
                "{a:014,.1%}",
                NativePrimitive::Number(1e+20),
                "10,000,000,000,000,000,000,000.0%",
            ),
            (
                "{a:08_.0f}",
                NativePrimitive::Number(1e+20),
                "100_000_000_000_000_000_000",
            ),
            (
                "{a:012_.0f}",
                NativePrimitive::Number(1e+20),
                "100_000_000_000_000_000_000",
            ),
            (
                "{a:014_.0f}",
                NativePrimitive::Number(1e+20),
                "100_000_000_000_000_000_000",
            ),
            (
                "{a:08_.2f}",
                NativePrimitive::Number(1e+20),
                "100_000_000_000_000_000_000.00",
            ),
            (
                "{a:012_.2f}",
                NativePrimitive::Number(1e+20),
                "100_000_000_000_000_000_000.00",
            ),
            (
                "{a:014_.2f}",
                NativePrimitive::Number(1e+20),
                "100_000_000_000_000_000_000.00",
            ),
            ("{a:08_.1e}", NativePrimitive::Number(1e+20), "01.0e+20"),
            (
                "{a:012_.1e}",
                NativePrimitive::Number(1e+20),
                "00_001.0e+20",
            ),
            (
                "{a:014_.1e}",
                NativePrimitive::Number(1e+20),
                "0_000_001.0e+20",
            ),
            ("{a:08_.3g}", NativePrimitive::Number(1e+20), "0_001e+20"),
            (
                "{a:012_.3g}",
                NativePrimitive::Number(1e+20),
                "0_000_001e+20",
            ),
            (
                "{a:014_.3g}",
                NativePrimitive::Number(1e+20),
                "00_000_001e+20",
            ),
            (
                "{a:08_.0%}",
                NativePrimitive::Number(1e+20),
                "10_000_000_000_000_000_000_000%",
            ),
            (
                "{a:012_.0%}",
                NativePrimitive::Number(1e+20),
                "10_000_000_000_000_000_000_000%",
            ),
            (
                "{a:014_.0%}",
                NativePrimitive::Number(1e+20),
                "10_000_000_000_000_000_000_000%",
            ),
            (
                "{a:08_.1%}",
                NativePrimitive::Number(1e+20),
                "10_000_000_000_000_000_000_000.0%",
            ),
            (
                "{a:012_.1%}",
                NativePrimitive::Number(1e+20),
                "10_000_000_000_000_000_000_000.0%",
            ),
            (
                "{a:014_.1%}",
                NativePrimitive::Number(1e+20),
                "10_000_000_000_000_000_000_000.0%",
            ),
            (
                "{a:+016,.2f}",
                NativePrimitive::Number(12345.5),
                "+0,000,012,345.50",
            ),
            (
                "{a:0=+016,.2f}",
                NativePrimitive::Number(12345.5),
                "+0,000,012,345.50",
            ),
            (
                "{a:0>+016,.2f}",
                NativePrimitive::Number(12345.5),
                "000000+12,345.50",
            ),
            (
                "{a:0<+016,.2f}",
                NativePrimitive::Number(12345.5),
                "+12,345.50000000",
            ),
            (
                "{a:0^+016,.2f}",
                NativePrimitive::Number(12345.5),
                "000+12,345.50000",
            ),
            (
                "{a:*=+016,.2f}",
                NativePrimitive::Number(12345.5),
                "+******12,345.50",
            ),
            (
                "{a:😀=+016,.2f}",
                NativePrimitive::Number(12345.5),
                "+😀😀😀😀😀😀12,345.50",
            ),
            (
                "{a:0=+014,.1e}",
                NativePrimitive::Number(12345.5),
                "+000,001.2e+04",
            ),
            ("{a:,.0%}", NativePrimitive::Number(12345.5), "1,234,550%"),
            ("{a:_.0%}", NativePrimitive::Number(12345.5), "1_234_550%"),
            (
                "{a:z016,.0f}",
                NativePrimitive::Number(12345.5),
                "0,000,000,012,346",
            ),
            (
                "{a:+016,.2f}",
                NativePrimitive::Number(-12345.5),
                "-0,000,012,345.50",
            ),
            (
                "{a:0=+016,.2f}",
                NativePrimitive::Number(-12345.5),
                "-0,000,012,345.50",
            ),
            (
                "{a:0>+016,.2f}",
                NativePrimitive::Number(-12345.5),
                "000000-12,345.50",
            ),
            (
                "{a:0<+016,.2f}",
                NativePrimitive::Number(-12345.5),
                "-12,345.50000000",
            ),
            (
                "{a:0^+016,.2f}",
                NativePrimitive::Number(-12345.5),
                "000-12,345.50000",
            ),
            (
                "{a:*=+016,.2f}",
                NativePrimitive::Number(-12345.5),
                "-******12,345.50",
            ),
            (
                "{a:😀=+016,.2f}",
                NativePrimitive::Number(-12345.5),
                "-😀😀😀😀😀😀12,345.50",
            ),
            (
                "{a:0=+014,.1e}",
                NativePrimitive::Number(-12345.5),
                "-000,001.2e+04",
            ),
            ("{a:,.0%}", NativePrimitive::Number(-12345.5), "-1,234,550%"),
            ("{a:_.0%}", NativePrimitive::Number(-12345.5), "-1_234_550%"),
            (
                "{a:z016,.0f}",
                NativePrimitive::Number(-12345.5),
                "-000,000,012,346",
            ),
            (
                "{a:+016,.2f}",
                NativePrimitive::Number(1.25),
                "+0,000,000,001.25",
            ),
            (
                "{a:0=+016,.2f}",
                NativePrimitive::Number(1.25),
                "+0,000,000,001.25",
            ),
            (
                "{a:0>+016,.2f}",
                NativePrimitive::Number(1.25),
                "00000000000+1.25",
            ),
            (
                "{a:0<+016,.2f}",
                NativePrimitive::Number(1.25),
                "+1.2500000000000",
            ),
            (
                "{a:0^+016,.2f}",
                NativePrimitive::Number(1.25),
                "00000+1.25000000",
            ),
            (
                "{a:*=+016,.2f}",
                NativePrimitive::Number(1.25),
                "+***********1.25",
            ),
            (
                "{a:😀=+016,.2f}",
                NativePrimitive::Number(1.25),
                "+😀😀😀😀😀😀😀😀😀😀😀1.25",
            ),
            (
                "{a:0=+014,.1e}",
                NativePrimitive::Number(1.25),
                "+000,001.2e+00",
            ),
            ("{a:,.0%}", NativePrimitive::Number(1.25), "125%"),
            ("{a:_.0%}", NativePrimitive::Number(1.25), "125%"),
            (
                "{a:z016,.0f}",
                NativePrimitive::Number(1.25),
                "0,000,000,000,001",
            ),
            (
                "{a:+016,.2f}",
                NativePrimitive::Number(-1.25),
                "-0,000,000,001.25",
            ),
            (
                "{a:0=+016,.2f}",
                NativePrimitive::Number(-1.25),
                "-0,000,000,001.25",
            ),
            (
                "{a:0>+016,.2f}",
                NativePrimitive::Number(-1.25),
                "00000000000-1.25",
            ),
            (
                "{a:0<+016,.2f}",
                NativePrimitive::Number(-1.25),
                "-1.2500000000000",
            ),
            (
                "{a:0^+016,.2f}",
                NativePrimitive::Number(-1.25),
                "00000-1.25000000",
            ),
            (
                "{a:*=+016,.2f}",
                NativePrimitive::Number(-1.25),
                "-***********1.25",
            ),
            (
                "{a:😀=+016,.2f}",
                NativePrimitive::Number(-1.25),
                "-😀😀😀😀😀😀😀😀😀😀😀1.25",
            ),
            (
                "{a:0=+014,.1e}",
                NativePrimitive::Number(-1.25),
                "-000,001.2e+00",
            ),
            ("{a:,.0%}", NativePrimitive::Number(-1.25), "-125%"),
            ("{a:_.0%}", NativePrimitive::Number(-1.25), "-125%"),
            (
                "{a:z016,.0f}",
                NativePrimitive::Number(-1.25),
                "-000,000,000,001",
            ),
            (
                "{a:+016,.2f}",
                NativePrimitive::Number(-0.0),
                "-0,000,000,000.00",
            ),
            (
                "{a:0=+016,.2f}",
                NativePrimitive::Number(-0.0),
                "-0,000,000,000.00",
            ),
            (
                "{a:0>+016,.2f}",
                NativePrimitive::Number(-0.0),
                "00000000000-0.00",
            ),
            (
                "{a:0<+016,.2f}",
                NativePrimitive::Number(-0.0),
                "-0.0000000000000",
            ),
            (
                "{a:0^+016,.2f}",
                NativePrimitive::Number(-0.0),
                "00000-0.00000000",
            ),
            (
                "{a:*=+016,.2f}",
                NativePrimitive::Number(-0.0),
                "-***********0.00",
            ),
            (
                "{a:😀=+016,.2f}",
                NativePrimitive::Number(-0.0),
                "-😀😀😀😀😀😀😀😀😀😀😀0.00",
            ),
            (
                "{a:0=+014,.1e}",
                NativePrimitive::Number(-0.0),
                "-000,000.0e+00",
            ),
            ("{a:,.0%}", NativePrimitive::Number(-0.0), "-0%"),
            ("{a:_.0%}", NativePrimitive::Number(-0.0), "-0%"),
            (
                "{a:z016,.0f}",
                NativePrimitive::Number(-0.0),
                "0,000,000,000,000",
            ),
        ] {
            let values = BTreeMap::from([("a".to_owned(), primitive(value))]);
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default())?,
                expected,
                "original grouped padding: {template}, {values:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn python_312_grouped_zero_padding_retains_explicit_alignment_and_nested_fields()
    -> Result<(), Box<dyn Error>> {
        for (template, value, expected) in [
            (
                "{a:014,d}",
                NativePrimitive::Integer(12345),
                "00,000,012,345",
            ),
            (
                "{a:014_d}",
                NativePrimitive::Integer(12345),
                "00_000_012_345",
            ),
            (
                "{a:=014,d}",
                NativePrimitive::Integer(12345),
                "00,000,012,345",
            ),
            (
                "{a:>014,d}",
                NativePrimitive::Integer(12345),
                "0000000012,345",
            ),
            (
                "{a:<014,d}",
                NativePrimitive::Integer(12345),
                "12,34500000000",
            ),
            (
                "{a:^014,d}",
                NativePrimitive::Integer(12345),
                "000012,3450000",
            ),
            (
                "{a:0>014,d}",
                NativePrimitive::Integer(12345),
                "0000000012,345",
            ),
            (
                "{a:0<014,d}",
                NativePrimitive::Integer(12345),
                "12,34500000000",
            ),
            (
                "{a:0^014,d}",
                NativePrimitive::Integer(12345),
                "000012,3450000",
            ),
            (
                "{a: >014,d}",
                NativePrimitive::Integer(12345),
                "        12,345",
            ),
            (
                "{a:*>014,d}",
                NativePrimitive::Integer(12345),
                "********12,345",
            ),
            (
                "{a:*=014,d}",
                NativePrimitive::Integer(12345),
                "********12,345",
            ),
            (
                "{a:*=+014,d}",
                NativePrimitive::Integer(12345),
                "+*******12,345",
            ),
            (
                "{a:0=+014,d}",
                NativePrimitive::Integer(12345),
                "+0,000,012,345",
            ),
            (
                "{a:0= 014,d}",
                NativePrimitive::Integer(12345),
                " 0,000,012,345",
            ),
            (
                "{a:014,d}",
                NativePrimitive::Integer(-12345),
                "-0,000,012,345",
            ),
            (
                "{a:014_d}",
                NativePrimitive::Integer(-12345),
                "-0_000_012_345",
            ),
            (
                "{a:=014,d}",
                NativePrimitive::Integer(-12345),
                "-0,000,012,345",
            ),
            (
                "{a:>014,d}",
                NativePrimitive::Integer(-12345),
                "0000000-12,345",
            ),
            (
                "{a:<014,d}",
                NativePrimitive::Integer(-12345),
                "-12,3450000000",
            ),
            (
                "{a:^014,d}",
                NativePrimitive::Integer(-12345),
                "000-12,3450000",
            ),
            (
                "{a:0>014,d}",
                NativePrimitive::Integer(-12345),
                "0000000-12,345",
            ),
            (
                "{a:0<014,d}",
                NativePrimitive::Integer(-12345),
                "-12,3450000000",
            ),
            (
                "{a:0^014,d}",
                NativePrimitive::Integer(-12345),
                "000-12,3450000",
            ),
            (
                "{a: >014,d}",
                NativePrimitive::Integer(-12345),
                "       -12,345",
            ),
            (
                "{a:*>014,d}",
                NativePrimitive::Integer(-12345),
                "*******-12,345",
            ),
            (
                "{a:*=014,d}",
                NativePrimitive::Integer(-12345),
                "-*******12,345",
            ),
            (
                "{a:*=+014,d}",
                NativePrimitive::Integer(-12345),
                "-*******12,345",
            ),
            (
                "{a:0=+014,d}",
                NativePrimitive::Integer(-12345),
                "-0,000,012,345",
            ),
            (
                "{a:0= 014,d}",
                NativePrimitive::Integer(-12345),
                "-0,000,012,345",
            ),
            (
                "{a!s:014s}",
                NativePrimitive::Integer(12345),
                "12345000000000",
            ),
            (
                "{a:😀=014,d}",
                NativePrimitive::Integer(-12345),
                "-😀😀😀😀😀😀😀12,345",
            ),
            (
                "{a:٠١٤,d}",
                NativePrimitive::Integer(12345),
                "        12,345",
            ),
        ] {
            let values = BTreeMap::from([("a".to_owned(), primitive(value))]);
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default())?,
                expected,
                "original grouped padding: {template}, {values:?}"
            );
        }
        let values = BTreeMap::from([(
            "a".to_owned(),
            NativeValue::PreservedUnknown {
                type_name: "JSON".to_owned(),
                value: serde_json::json!(u64::MAX),
            },
        )]);
        assert_eq!(
            NativeTextFormatter::format("{a:014,d}", &values, &CancellationToken::default())?,
            "18,446,744,073,709,551,615"
        );
        let values = BTreeMap::from([
            ("a".to_owned(), primitive(NativePrimitive::Integer(-12_345))),
            (
                "b".to_owned(),
                primitive(NativePrimitive::String("١４".to_owned())),
            ),
        ]);
        assert_eq!(
            NativeTextFormatter::format("{a:0=+0{b},d}", &values, &CancellationToken::default())?,
            "-0,000,012,345"
        );
        let values = BTreeMap::from([
            (
                "a".to_owned(),
                NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value: serde_json::json!({"int": -7}),
                },
            ),
            (
                "b".to_owned(),
                primitive(NativePrimitive::String("١０".to_owned())),
            ),
        ]);
        assert_eq!(
            NativeTextFormatter::format(
                "{a[int]:0=+#0{b}_x}",
                &values,
                &CancellationToken::default()
            )?,
            "-0x00_0007"
        );
        Ok(())
    }

    #[test]
    fn python_312_grouped_zero_padding_checks_separator_growth_before_allocation()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Integer(7)))]);
        for template in [
            "{a:016777216,d}",
            "{a:016777217,d}",
            "{a:0=+#016777217_x}",
            "{a:😀=016777216,d}",
        ] {
            assert!(
                matches!(
                    NativeTextFormatter::format(template, &values, &CancellationToken::default()),
                    Err(NativeTextFormatError::ResultTooLarge { .. })
                ),
                "{template}"
            );
        }
        for (template, expected) in [
            (
                "{a:016777217,q}",
                "grouping is unsupported for this format type",
            ),
            (
                "{a:016777217,_d}",
                "cannot combine comma and underscore grouping",
            ),
            ("{a:016777217,.2d}", "integer precision is unsupported"),
        ] {
            assert!(
                matches!(NativeTextFormatter::format(template, &values, &CancellationToken::default()), Err(NativeTextFormatError::InvalidTemplate(error)) if error == expected),
                "{template}"
            );
        }
        assert!(matches!(
            group_digits("1", '😀', 3, NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES, 0),
            Err(NativeTextFormatError::ResultTooLarge { .. })
        ));
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert!(matches!(
            NativeTextFormatter::format("{a:016777216,d}", &values, &cancellation),
            Err(NativeTextFormatError::Cancelled)
        ));
        assert_eq!(
            NativeTextFormatter::format("{a:08,d}", &values, &CancellationToken::default())?,
            "0,000,007"
        );
        Ok(())
    }

    #[test]
    fn python_312_integer_float_presentations_round_binary64_boundaries()
    -> Result<(), Box<dyn Error>> {
        for (template, value, expected) in [
            (
                "{a:.17f}",
                NativePrimitive::Boolean(false),
                "0.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Boolean(false),
                "0.00000000000000000e+00",
            ),
            ("{a:.17g}", NativePrimitive::Boolean(false), "0"),
            (
                "{a:.17f}",
                NativePrimitive::Boolean(true),
                "1.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Boolean(true),
                "1.00000000000000000e+00",
            ),
            ("{a:.17g}", NativePrimitive::Boolean(true), "1"),
            (
                "{a:.17f}",
                NativePrimitive::Integer(0),
                "0.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(0),
                "0.00000000000000000e+00",
            ),
            ("{a:.17g}", NativePrimitive::Integer(0), "0"),
            (
                "{a:.17f}",
                NativePrimitive::Integer(1),
                "1.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(1),
                "1.00000000000000000e+00",
            ),
            ("{a:.17g}", NativePrimitive::Integer(1), "1"),
            (
                "{a:.17f}",
                NativePrimitive::Integer(-1),
                "-1.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(-1),
                "-1.00000000000000000e+00",
            ),
            ("{a:.17g}", NativePrimitive::Integer(-1), "-1"),
            (
                "{a:.17f}",
                NativePrimitive::Integer(7),
                "7.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(7),
                "7.00000000000000000e+00",
            ),
            ("{a:.17g}", NativePrimitive::Integer(7), "7"),
            (
                "{a:.17f}",
                NativePrimitive::Integer(-7),
                "-7.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(-7),
                "-7.00000000000000000e+00",
            ),
            ("{a:.17g}", NativePrimitive::Integer(-7), "-7"),
            (
                "{a:.17f}",
                NativePrimitive::Integer(65),
                "65.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(65),
                "6.50000000000000000e+01",
            ),
            ("{a:.17g}", NativePrimitive::Integer(65), "65"),
            (
                "{a:.17f}",
                NativePrimitive::Integer(999),
                "999.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(999),
                "9.99000000000000000e+02",
            ),
            ("{a:.17g}", NativePrimitive::Integer(999), "999"),
            (
                "{a:.17f}",
                NativePrimitive::Integer(9995),
                "9995.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(9995),
                "9.99500000000000000e+03",
            ),
            ("{a:.17g}", NativePrimitive::Integer(9995), "9995"),
            (
                "{a:.17f}",
                NativePrimitive::Integer(99995),
                "99995.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(99995),
                "9.99950000000000000e+04",
            ),
            ("{a:.17g}", NativePrimitive::Integer(99995), "99995"),
            (
                "{a:.17f}",
                NativePrimitive::Integer(9007199254740989),
                "9007199254740989.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(9007199254740989),
                "9.00719925474098900e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(9007199254740989),
                "9007199254740989",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(9007199254740990),
                "9007199254740990.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(9007199254740990),
                "9.00719925474099000e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(9007199254740990),
                "9007199254740990",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(9007199254740991),
                "9007199254740991.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(9007199254740991),
                "9.00719925474099100e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(9007199254740991),
                "9007199254740991",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(9007199254740992),
                "9007199254740992.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(9007199254740992),
                "9.00719925474099200e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(9007199254740992),
                "9007199254740992",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(9007199254740993),
                "9007199254740992.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(9007199254740993),
                "9.00719925474099200e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(9007199254740993),
                "9007199254740992",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(9007199254740994),
                "9007199254740994.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(9007199254740994),
                "9.00719925474099400e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(9007199254740994),
                "9007199254740994",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(9007199254740995),
                "9007199254740996.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(9007199254740995),
                "9.00719925474099600e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(9007199254740995),
                "9007199254740996",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(9007199254740996),
                "9007199254740996.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(9007199254740996),
                "9.00719925474099600e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(9007199254740996),
                "9007199254740996",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(9007199254740997),
                "9007199254740996.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(9007199254740997),
                "9.00719925474099600e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(9007199254740997),
                "9007199254740996",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(9007199254740998),
                "9007199254740998.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(9007199254740998),
                "9.00719925474099800e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(9007199254740998),
                "9007199254740998",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(9007199254740999),
                "9007199254741000.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(9007199254740999),
                "9.00719925474100000e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(9007199254740999),
                "9007199254741000",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(9007199254741000),
                "9007199254741000.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(9007199254741000),
                "9.00719925474100000e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(9007199254741000),
                "9007199254741000",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(-9007199254740989),
                "-9007199254740989.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(-9007199254740989),
                "-9.00719925474098900e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(-9007199254740989),
                "-9007199254740989",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(-9007199254740990),
                "-9007199254740990.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(-9007199254740990),
                "-9.00719925474099000e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(-9007199254740990),
                "-9007199254740990",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(-9007199254740991),
                "-9007199254740991.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(-9007199254740991),
                "-9.00719925474099100e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(-9007199254740991),
                "-9007199254740991",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(-9007199254740992),
                "-9007199254740992.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(-9007199254740992),
                "-9.00719925474099200e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(-9007199254740992),
                "-9007199254740992",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(-9007199254740993),
                "-9007199254740992.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(-9007199254740993),
                "-9.00719925474099200e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(-9007199254740993),
                "-9007199254740992",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(-9007199254740994),
                "-9007199254740994.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(-9007199254740994),
                "-9.00719925474099400e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(-9007199254740994),
                "-9007199254740994",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(-9007199254740995),
                "-9007199254740996.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(-9007199254740995),
                "-9.00719925474099600e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(-9007199254740995),
                "-9007199254740996",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(-9007199254740996),
                "-9007199254740996.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(-9007199254740996),
                "-9.00719925474099600e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(-9007199254740996),
                "-9007199254740996",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(-9007199254740997),
                "-9007199254740996.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(-9007199254740997),
                "-9.00719925474099600e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(-9007199254740997),
                "-9007199254740996",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(-9007199254740998),
                "-9007199254740998.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(-9007199254740998),
                "-9.00719925474099800e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(-9007199254740998),
                "-9007199254740998",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(-9007199254740999),
                "-9007199254741000.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(-9007199254740999),
                "-9.00719925474100000e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(-9007199254740999),
                "-9007199254741000",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(-9007199254741000),
                "-9007199254741000.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(-9007199254741000),
                "-9.00719925474100000e+15",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(-9007199254741000),
                "-9007199254741000",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9223372036854775808.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9.22337203685477581e+18",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9.2233720368547758e+18",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(9223372036854775807),
                "9223372036854775808.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(9223372036854775807),
                "9.22337203685477581e+18",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(9223372036854775807),
                "9.2233720368547758e+18",
            ),
            (
                "{a:.17f}",
                NativePrimitive::Integer(9223372036854775295),
                "9223372036854774784.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::Integer(9223372036854775295),
                "9.22337203685477478e+18",
            ),
            (
                "{a:.17g}",
                NativePrimitive::Integer(9223372036854775295),
                "9.2233720368547748e+18",
            ),
            (
                "{a:.17f}",
                NativePrimitive::UnsignedInteger(9223372036854776832),
                "9223372036854775808.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::UnsignedInteger(9223372036854776832),
                "9.22337203685477581e+18",
            ),
            (
                "{a:.17g}",
                NativePrimitive::UnsignedInteger(9223372036854776832),
                "9.2233720368547758e+18",
            ),
            (
                "{a:.17f}",
                NativePrimitive::UnsignedInteger(9223372036854778880),
                "9223372036854779904.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::UnsignedInteger(9223372036854778880),
                "9.22337203685477990e+18",
            ),
            (
                "{a:.17g}",
                NativePrimitive::UnsignedInteger(9223372036854778880),
                "9.2233720368547799e+18",
            ),
            (
                "{a:.17f}",
                NativePrimitive::UnsignedInteger(18446744073709551615),
                "18446744073709551616.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::UnsignedInteger(18446744073709551615),
                "1.84467440737095516e+19",
            ),
            (
                "{a:.17g}",
                NativePrimitive::UnsignedInteger(18446744073709551615),
                "1.8446744073709552e+19",
            ),
            (
                "{a:.17f}",
                NativePrimitive::UnsignedInteger(18446744073709550591),
                "18446744073709549568.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::UnsignedInteger(18446744073709550591),
                "1.84467440737095496e+19",
            ),
            (
                "{a:.17g}",
                NativePrimitive::UnsignedInteger(18446744073709550591),
                "1.844674407370955e+19",
            ),
            (
                "{a:.17f}",
                NativePrimitive::UnsignedInteger(18446744073709549567),
                "18446744073709549568.00000000000000000",
            ),
            (
                "{a:.17e}",
                NativePrimitive::UnsignedInteger(18446744073709549567),
                "1.84467440737095496e+19",
            ),
            (
                "{a:.17g}",
                NativePrimitive::UnsignedInteger(18446744073709549567),
                "1.844674407370955e+19",
            ),
        ] {
            let values = BTreeMap::from([("a".to_owned(), primitive(value))]);
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default())?,
                expected,
                "original integer float presentation {template}, {values:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn python_312_integer_float_presentations_reuse_sign_zero_alternate_and_grouping()
    -> Result<(), Box<dyn Error>> {
        for (template, value, expected) in [
            ("{a:f}", NativePrimitive::Boolean(false), "0.000000"),
            ("{a:.0f}", NativePrimitive::Boolean(false), "0"),
            ("{a:z.0f}", NativePrimitive::Boolean(false), "0"),
            ("{a:#.0f}", NativePrimitive::Boolean(false), "0."),
            (
                "{a:+z#014,.2f}",
                NativePrimitive::Boolean(false),
                "+00,000,000.00",
            ),
            (
                "{a:0=+018_.2f}",
                NativePrimitive::Boolean(false),
                "+00_000_000_000.00",
            ),
            (
                "{a:*=+018,.2f}",
                NativePrimitive::Boolean(false),
                "+*************0.00",
            ),
            (
                "{a:>018,.2f}",
                NativePrimitive::Boolean(false),
                "000000000000000.00",
            ),
            (
                "{a:^018_.2f}",
                NativePrimitive::Boolean(false),
                "00000000.000000000",
            ),
            ("{a:F}", NativePrimitive::Boolean(false), "0.000000"),
            ("{a:.0F}", NativePrimitive::Boolean(false), "0"),
            ("{a:z.0F}", NativePrimitive::Boolean(false), "0"),
            ("{a:#.0F}", NativePrimitive::Boolean(false), "0."),
            (
                "{a:+z#014,.2F}",
                NativePrimitive::Boolean(false),
                "+00,000,000.00",
            ),
            (
                "{a:0=+018_.2F}",
                NativePrimitive::Boolean(false),
                "+00_000_000_000.00",
            ),
            (
                "{a:*=+018,.2F}",
                NativePrimitive::Boolean(false),
                "+*************0.00",
            ),
            (
                "{a:>018,.2F}",
                NativePrimitive::Boolean(false),
                "000000000000000.00",
            ),
            (
                "{a:^018_.2F}",
                NativePrimitive::Boolean(false),
                "00000000.000000000",
            ),
            ("{a:e}", NativePrimitive::Boolean(false), "0.000000e+00"),
            ("{a:.0e}", NativePrimitive::Boolean(false), "0e+00"),
            ("{a:z.0e}", NativePrimitive::Boolean(false), "0e+00"),
            ("{a:#.0e}", NativePrimitive::Boolean(false), "0.e+00"),
            (
                "{a:+z#014,.2e}",
                NativePrimitive::Boolean(false),
                "+00,000.00e+00",
            ),
            (
                "{a:0=+018_.2e}",
                NativePrimitive::Boolean(false),
                "+00_000_000.00e+00",
            ),
            (
                "{a:*=+018,.2e}",
                NativePrimitive::Boolean(false),
                "+*********0.00e+00",
            ),
            (
                "{a:>018,.2e}",
                NativePrimitive::Boolean(false),
                "00000000000.00e+00",
            ),
            (
                "{a:^018_.2e}",
                NativePrimitive::Boolean(false),
                "000000.00e+0000000",
            ),
            ("{a:E}", NativePrimitive::Boolean(false), "0.000000E+00"),
            ("{a:.0E}", NativePrimitive::Boolean(false), "0E+00"),
            ("{a:z.0E}", NativePrimitive::Boolean(false), "0E+00"),
            ("{a:#.0E}", NativePrimitive::Boolean(false), "0.E+00"),
            (
                "{a:+z#014,.2E}",
                NativePrimitive::Boolean(false),
                "+00,000.00E+00",
            ),
            (
                "{a:0=+018_.2E}",
                NativePrimitive::Boolean(false),
                "+00_000_000.00E+00",
            ),
            (
                "{a:*=+018,.2E}",
                NativePrimitive::Boolean(false),
                "+*********0.00E+00",
            ),
            (
                "{a:>018,.2E}",
                NativePrimitive::Boolean(false),
                "00000000000.00E+00",
            ),
            (
                "{a:^018_.2E}",
                NativePrimitive::Boolean(false),
                "000000.00E+0000000",
            ),
            ("{a:g}", NativePrimitive::Boolean(false), "0"),
            ("{a:.0g}", NativePrimitive::Boolean(false), "0"),
            ("{a:z.0g}", NativePrimitive::Boolean(false), "0"),
            ("{a:#.0g}", NativePrimitive::Boolean(false), "0."),
            (
                "{a:+z#014,.2g}",
                NativePrimitive::Boolean(false),
                "+000,000,000.0",
            ),
            (
                "{a:0=+018_.2g}",
                NativePrimitive::Boolean(false),
                "+0_000_000_000_000",
            ),
            (
                "{a:*=+018,.2g}",
                NativePrimitive::Boolean(false),
                "+****************0",
            ),
            (
                "{a:>018,.2g}",
                NativePrimitive::Boolean(false),
                "000000000000000000",
            ),
            (
                "{a:^018_.2g}",
                NativePrimitive::Boolean(false),
                "000000000000000000",
            ),
            ("{a:G}", NativePrimitive::Boolean(false), "0"),
            ("{a:.0G}", NativePrimitive::Boolean(false), "0"),
            ("{a:z.0G}", NativePrimitive::Boolean(false), "0"),
            ("{a:#.0G}", NativePrimitive::Boolean(false), "0."),
            (
                "{a:+z#014,.2G}",
                NativePrimitive::Boolean(false),
                "+000,000,000.0",
            ),
            (
                "{a:0=+018_.2G}",
                NativePrimitive::Boolean(false),
                "+0_000_000_000_000",
            ),
            (
                "{a:*=+018,.2G}",
                NativePrimitive::Boolean(false),
                "+****************0",
            ),
            (
                "{a:>018,.2G}",
                NativePrimitive::Boolean(false),
                "000000000000000000",
            ),
            (
                "{a:^018_.2G}",
                NativePrimitive::Boolean(false),
                "000000000000000000",
            ),
            ("{a:%}", NativePrimitive::Boolean(false), "0.000000%"),
            ("{a:.0%}", NativePrimitive::Boolean(false), "0%"),
            ("{a:z.0%}", NativePrimitive::Boolean(false), "0%"),
            ("{a:#.0%}", NativePrimitive::Boolean(false), "0.%"),
            (
                "{a:+z#014,.2%}",
                NativePrimitive::Boolean(false),
                "+0,000,000.00%",
            ),
            (
                "{a:0=+018_.2%}",
                NativePrimitive::Boolean(false),
                "+0_000_000_000.00%",
            ),
            (
                "{a:*=+018,.2%}",
                NativePrimitive::Boolean(false),
                "+************0.00%",
            ),
            (
                "{a:>018,.2%}",
                NativePrimitive::Boolean(false),
                "00000000000000.00%",
            ),
            (
                "{a:^018_.2%}",
                NativePrimitive::Boolean(false),
                "0000000.00%0000000",
            ),
            ("{a:f}", NativePrimitive::Boolean(true), "1.000000"),
            ("{a:.0f}", NativePrimitive::Boolean(true), "1"),
            ("{a:z.0f}", NativePrimitive::Boolean(true), "1"),
            ("{a:#.0f}", NativePrimitive::Boolean(true), "1."),
            (
                "{a:+z#014,.2f}",
                NativePrimitive::Boolean(true),
                "+00,000,001.00",
            ),
            (
                "{a:0=+018_.2f}",
                NativePrimitive::Boolean(true),
                "+00_000_000_001.00",
            ),
            (
                "{a:*=+018,.2f}",
                NativePrimitive::Boolean(true),
                "+*************1.00",
            ),
            (
                "{a:>018,.2f}",
                NativePrimitive::Boolean(true),
                "000000000000001.00",
            ),
            (
                "{a:^018_.2f}",
                NativePrimitive::Boolean(true),
                "00000001.000000000",
            ),
            ("{a:F}", NativePrimitive::Boolean(true), "1.000000"),
            ("{a:.0F}", NativePrimitive::Boolean(true), "1"),
            ("{a:z.0F}", NativePrimitive::Boolean(true), "1"),
            ("{a:#.0F}", NativePrimitive::Boolean(true), "1."),
            (
                "{a:+z#014,.2F}",
                NativePrimitive::Boolean(true),
                "+00,000,001.00",
            ),
            (
                "{a:0=+018_.2F}",
                NativePrimitive::Boolean(true),
                "+00_000_000_001.00",
            ),
            (
                "{a:*=+018,.2F}",
                NativePrimitive::Boolean(true),
                "+*************1.00",
            ),
            (
                "{a:>018,.2F}",
                NativePrimitive::Boolean(true),
                "000000000000001.00",
            ),
            (
                "{a:^018_.2F}",
                NativePrimitive::Boolean(true),
                "00000001.000000000",
            ),
            ("{a:e}", NativePrimitive::Boolean(true), "1.000000e+00"),
            ("{a:.0e}", NativePrimitive::Boolean(true), "1e+00"),
            ("{a:z.0e}", NativePrimitive::Boolean(true), "1e+00"),
            ("{a:#.0e}", NativePrimitive::Boolean(true), "1.e+00"),
            (
                "{a:+z#014,.2e}",
                NativePrimitive::Boolean(true),
                "+00,001.00e+00",
            ),
            (
                "{a:0=+018_.2e}",
                NativePrimitive::Boolean(true),
                "+00_000_001.00e+00",
            ),
            (
                "{a:*=+018,.2e}",
                NativePrimitive::Boolean(true),
                "+*********1.00e+00",
            ),
            (
                "{a:>018,.2e}",
                NativePrimitive::Boolean(true),
                "00000000001.00e+00",
            ),
            (
                "{a:^018_.2e}",
                NativePrimitive::Boolean(true),
                "000001.00e+0000000",
            ),
            ("{a:E}", NativePrimitive::Boolean(true), "1.000000E+00"),
            ("{a:.0E}", NativePrimitive::Boolean(true), "1E+00"),
            ("{a:z.0E}", NativePrimitive::Boolean(true), "1E+00"),
            ("{a:#.0E}", NativePrimitive::Boolean(true), "1.E+00"),
            (
                "{a:+z#014,.2E}",
                NativePrimitive::Boolean(true),
                "+00,001.00E+00",
            ),
            (
                "{a:0=+018_.2E}",
                NativePrimitive::Boolean(true),
                "+00_000_001.00E+00",
            ),
            (
                "{a:*=+018,.2E}",
                NativePrimitive::Boolean(true),
                "+*********1.00E+00",
            ),
            (
                "{a:>018,.2E}",
                NativePrimitive::Boolean(true),
                "00000000001.00E+00",
            ),
            (
                "{a:^018_.2E}",
                NativePrimitive::Boolean(true),
                "000001.00E+0000000",
            ),
            ("{a:g}", NativePrimitive::Boolean(true), "1"),
            ("{a:.0g}", NativePrimitive::Boolean(true), "1"),
            ("{a:z.0g}", NativePrimitive::Boolean(true), "1"),
            ("{a:#.0g}", NativePrimitive::Boolean(true), "1."),
            (
                "{a:+z#014,.2g}",
                NativePrimitive::Boolean(true),
                "+000,000,001.0",
            ),
            (
                "{a:0=+018_.2g}",
                NativePrimitive::Boolean(true),
                "+0_000_000_000_001",
            ),
            (
                "{a:*=+018,.2g}",
                NativePrimitive::Boolean(true),
                "+****************1",
            ),
            (
                "{a:>018,.2g}",
                NativePrimitive::Boolean(true),
                "000000000000000001",
            ),
            (
                "{a:^018_.2g}",
                NativePrimitive::Boolean(true),
                "000000001000000000",
            ),
            ("{a:G}", NativePrimitive::Boolean(true), "1"),
            ("{a:.0G}", NativePrimitive::Boolean(true), "1"),
            ("{a:z.0G}", NativePrimitive::Boolean(true), "1"),
            ("{a:#.0G}", NativePrimitive::Boolean(true), "1."),
            (
                "{a:+z#014,.2G}",
                NativePrimitive::Boolean(true),
                "+000,000,001.0",
            ),
            (
                "{a:0=+018_.2G}",
                NativePrimitive::Boolean(true),
                "+0_000_000_000_001",
            ),
            (
                "{a:*=+018,.2G}",
                NativePrimitive::Boolean(true),
                "+****************1",
            ),
            (
                "{a:>018,.2G}",
                NativePrimitive::Boolean(true),
                "000000000000000001",
            ),
            (
                "{a:^018_.2G}",
                NativePrimitive::Boolean(true),
                "000000001000000000",
            ),
            ("{a:%}", NativePrimitive::Boolean(true), "100.000000%"),
            ("{a:.0%}", NativePrimitive::Boolean(true), "100%"),
            ("{a:z.0%}", NativePrimitive::Boolean(true), "100%"),
            ("{a:#.0%}", NativePrimitive::Boolean(true), "100.%"),
            (
                "{a:+z#014,.2%}",
                NativePrimitive::Boolean(true),
                "+0,000,100.00%",
            ),
            (
                "{a:0=+018_.2%}",
                NativePrimitive::Boolean(true),
                "+0_000_000_100.00%",
            ),
            (
                "{a:*=+018,.2%}",
                NativePrimitive::Boolean(true),
                "+**********100.00%",
            ),
            (
                "{a:>018,.2%}",
                NativePrimitive::Boolean(true),
                "00000000000100.00%",
            ),
            (
                "{a:^018_.2%}",
                NativePrimitive::Boolean(true),
                "00000100.00%000000",
            ),
            ("{a:f}", NativePrimitive::Integer(-7), "-7.000000"),
            ("{a:.0f}", NativePrimitive::Integer(-7), "-7"),
            ("{a:z.0f}", NativePrimitive::Integer(-7), "-7"),
            ("{a:#.0f}", NativePrimitive::Integer(-7), "-7."),
            (
                "{a:+z#014,.2f}",
                NativePrimitive::Integer(-7),
                "-00,000,007.00",
            ),
            (
                "{a:0=+018_.2f}",
                NativePrimitive::Integer(-7),
                "-00_000_000_007.00",
            ),
            (
                "{a:*=+018,.2f}",
                NativePrimitive::Integer(-7),
                "-*************7.00",
            ),
            (
                "{a:>018,.2f}",
                NativePrimitive::Integer(-7),
                "0000000000000-7.00",
            ),
            (
                "{a:^018_.2f}",
                NativePrimitive::Integer(-7),
                "000000-7.000000000",
            ),
            ("{a:F}", NativePrimitive::Integer(-7), "-7.000000"),
            ("{a:.0F}", NativePrimitive::Integer(-7), "-7"),
            ("{a:z.0F}", NativePrimitive::Integer(-7), "-7"),
            ("{a:#.0F}", NativePrimitive::Integer(-7), "-7."),
            (
                "{a:+z#014,.2F}",
                NativePrimitive::Integer(-7),
                "-00,000,007.00",
            ),
            (
                "{a:0=+018_.2F}",
                NativePrimitive::Integer(-7),
                "-00_000_000_007.00",
            ),
            (
                "{a:*=+018,.2F}",
                NativePrimitive::Integer(-7),
                "-*************7.00",
            ),
            (
                "{a:>018,.2F}",
                NativePrimitive::Integer(-7),
                "0000000000000-7.00",
            ),
            (
                "{a:^018_.2F}",
                NativePrimitive::Integer(-7),
                "000000-7.000000000",
            ),
            ("{a:e}", NativePrimitive::Integer(-7), "-7.000000e+00"),
            ("{a:.0e}", NativePrimitive::Integer(-7), "-7e+00"),
            ("{a:z.0e}", NativePrimitive::Integer(-7), "-7e+00"),
            ("{a:#.0e}", NativePrimitive::Integer(-7), "-7.e+00"),
            (
                "{a:+z#014,.2e}",
                NativePrimitive::Integer(-7),
                "-00,007.00e+00",
            ),
            (
                "{a:0=+018_.2e}",
                NativePrimitive::Integer(-7),
                "-00_000_007.00e+00",
            ),
            (
                "{a:*=+018,.2e}",
                NativePrimitive::Integer(-7),
                "-*********7.00e+00",
            ),
            (
                "{a:>018,.2e}",
                NativePrimitive::Integer(-7),
                "000000000-7.00e+00",
            ),
            (
                "{a:^018_.2e}",
                NativePrimitive::Integer(-7),
                "0000-7.00e+0000000",
            ),
            ("{a:E}", NativePrimitive::Integer(-7), "-7.000000E+00"),
            ("{a:.0E}", NativePrimitive::Integer(-7), "-7E+00"),
            ("{a:z.0E}", NativePrimitive::Integer(-7), "-7E+00"),
            ("{a:#.0E}", NativePrimitive::Integer(-7), "-7.E+00"),
            (
                "{a:+z#014,.2E}",
                NativePrimitive::Integer(-7),
                "-00,007.00E+00",
            ),
            (
                "{a:0=+018_.2E}",
                NativePrimitive::Integer(-7),
                "-00_000_007.00E+00",
            ),
            (
                "{a:*=+018,.2E}",
                NativePrimitive::Integer(-7),
                "-*********7.00E+00",
            ),
            (
                "{a:>018,.2E}",
                NativePrimitive::Integer(-7),
                "000000000-7.00E+00",
            ),
            (
                "{a:^018_.2E}",
                NativePrimitive::Integer(-7),
                "0000-7.00E+0000000",
            ),
            ("{a:g}", NativePrimitive::Integer(-7), "-7"),
            ("{a:.0g}", NativePrimitive::Integer(-7), "-7"),
            ("{a:z.0g}", NativePrimitive::Integer(-7), "-7"),
            ("{a:#.0g}", NativePrimitive::Integer(-7), "-7."),
            (
                "{a:+z#014,.2g}",
                NativePrimitive::Integer(-7),
                "-000,000,007.0",
            ),
            (
                "{a:0=+018_.2g}",
                NativePrimitive::Integer(-7),
                "-0_000_000_000_007",
            ),
            (
                "{a:*=+018,.2g}",
                NativePrimitive::Integer(-7),
                "-****************7",
            ),
            (
                "{a:>018,.2g}",
                NativePrimitive::Integer(-7),
                "0000000000000000-7",
            ),
            (
                "{a:^018_.2g}",
                NativePrimitive::Integer(-7),
                "00000000-700000000",
            ),
            ("{a:G}", NativePrimitive::Integer(-7), "-7"),
            ("{a:.0G}", NativePrimitive::Integer(-7), "-7"),
            ("{a:z.0G}", NativePrimitive::Integer(-7), "-7"),
            ("{a:#.0G}", NativePrimitive::Integer(-7), "-7."),
            (
                "{a:+z#014,.2G}",
                NativePrimitive::Integer(-7),
                "-000,000,007.0",
            ),
            (
                "{a:0=+018_.2G}",
                NativePrimitive::Integer(-7),
                "-0_000_000_000_007",
            ),
            (
                "{a:*=+018,.2G}",
                NativePrimitive::Integer(-7),
                "-****************7",
            ),
            (
                "{a:>018,.2G}",
                NativePrimitive::Integer(-7),
                "0000000000000000-7",
            ),
            (
                "{a:^018_.2G}",
                NativePrimitive::Integer(-7),
                "00000000-700000000",
            ),
            ("{a:%}", NativePrimitive::Integer(-7), "-700.000000%"),
            ("{a:.0%}", NativePrimitive::Integer(-7), "-700%"),
            ("{a:z.0%}", NativePrimitive::Integer(-7), "-700%"),
            ("{a:#.0%}", NativePrimitive::Integer(-7), "-700.%"),
            (
                "{a:+z#014,.2%}",
                NativePrimitive::Integer(-7),
                "-0,000,700.00%",
            ),
            (
                "{a:0=+018_.2%}",
                NativePrimitive::Integer(-7),
                "-0_000_000_700.00%",
            ),
            (
                "{a:*=+018,.2%}",
                NativePrimitive::Integer(-7),
                "-**********700.00%",
            ),
            (
                "{a:>018,.2%}",
                NativePrimitive::Integer(-7),
                "0000000000-700.00%",
            ),
            (
                "{a:^018_.2%}",
                NativePrimitive::Integer(-7),
                "00000-700.00%00000",
            ),
            ("{a:f}", NativePrimitive::Integer(9995), "9995.000000"),
            ("{a:.0f}", NativePrimitive::Integer(9995), "9995"),
            ("{a:z.0f}", NativePrimitive::Integer(9995), "9995"),
            ("{a:#.0f}", NativePrimitive::Integer(9995), "9995."),
            (
                "{a:+z#014,.2f}",
                NativePrimitive::Integer(9995),
                "+00,009,995.00",
            ),
            (
                "{a:0=+018_.2f}",
                NativePrimitive::Integer(9995),
                "+00_000_009_995.00",
            ),
            (
                "{a:*=+018,.2f}",
                NativePrimitive::Integer(9995),
                "+*********9,995.00",
            ),
            (
                "{a:>018,.2f}",
                NativePrimitive::Integer(9995),
                "00000000009,995.00",
            ),
            (
                "{a:^018_.2f}",
                NativePrimitive::Integer(9995),
                "000009_995.0000000",
            ),
            ("{a:F}", NativePrimitive::Integer(9995), "9995.000000"),
            ("{a:.0F}", NativePrimitive::Integer(9995), "9995"),
            ("{a:z.0F}", NativePrimitive::Integer(9995), "9995"),
            ("{a:#.0F}", NativePrimitive::Integer(9995), "9995."),
            (
                "{a:+z#014,.2F}",
                NativePrimitive::Integer(9995),
                "+00,009,995.00",
            ),
            (
                "{a:0=+018_.2F}",
                NativePrimitive::Integer(9995),
                "+00_000_009_995.00",
            ),
            (
                "{a:*=+018,.2F}",
                NativePrimitive::Integer(9995),
                "+*********9,995.00",
            ),
            (
                "{a:>018,.2F}",
                NativePrimitive::Integer(9995),
                "00000000009,995.00",
            ),
            (
                "{a:^018_.2F}",
                NativePrimitive::Integer(9995),
                "000009_995.0000000",
            ),
            ("{a:e}", NativePrimitive::Integer(9995), "9.995000e+03"),
            ("{a:.0e}", NativePrimitive::Integer(9995), "1e+04"),
            ("{a:z.0e}", NativePrimitive::Integer(9995), "1e+04"),
            ("{a:#.0e}", NativePrimitive::Integer(9995), "1.e+04"),
            (
                "{a:+z#014,.2e}",
                NativePrimitive::Integer(9995),
                "+00,001.00e+04",
            ),
            (
                "{a:0=+018_.2e}",
                NativePrimitive::Integer(9995),
                "+00_000_001.00e+04",
            ),
            (
                "{a:*=+018,.2e}",
                NativePrimitive::Integer(9995),
                "+*********1.00e+04",
            ),
            (
                "{a:>018,.2e}",
                NativePrimitive::Integer(9995),
                "00000000001.00e+04",
            ),
            (
                "{a:^018_.2e}",
                NativePrimitive::Integer(9995),
                "000001.00e+0400000",
            ),
            ("{a:E}", NativePrimitive::Integer(9995), "9.995000E+03"),
            ("{a:.0E}", NativePrimitive::Integer(9995), "1E+04"),
            ("{a:z.0E}", NativePrimitive::Integer(9995), "1E+04"),
            ("{a:#.0E}", NativePrimitive::Integer(9995), "1.E+04"),
            (
                "{a:+z#014,.2E}",
                NativePrimitive::Integer(9995),
                "+00,001.00E+04",
            ),
            (
                "{a:0=+018_.2E}",
                NativePrimitive::Integer(9995),
                "+00_000_001.00E+04",
            ),
            (
                "{a:*=+018,.2E}",
                NativePrimitive::Integer(9995),
                "+*********1.00E+04",
            ),
            (
                "{a:>018,.2E}",
                NativePrimitive::Integer(9995),
                "00000000001.00E+04",
            ),
            (
                "{a:^018_.2E}",
                NativePrimitive::Integer(9995),
                "000001.00E+0400000",
            ),
            ("{a:g}", NativePrimitive::Integer(9995), "9995"),
            ("{a:.0g}", NativePrimitive::Integer(9995), "1e+04"),
            ("{a:z.0g}", NativePrimitive::Integer(9995), "1e+04"),
            ("{a:#.0g}", NativePrimitive::Integer(9995), "1.e+04"),
            (
                "{a:+z#014,.2g}",
                NativePrimitive::Integer(9995),
                "+000,001.0e+04",
            ),
            (
                "{a:0=+018_.2g}",
                NativePrimitive::Integer(9995),
                "+0_000_000_001e+04",
            ),
            (
                "{a:*=+018,.2g}",
                NativePrimitive::Integer(9995),
                "+************1e+04",
            ),
            (
                "{a:>018,.2g}",
                NativePrimitive::Integer(9995),
                "00000000000001e+04",
            ),
            (
                "{a:^018_.2g}",
                NativePrimitive::Integer(9995),
                "0000001e+040000000",
            ),
            ("{a:G}", NativePrimitive::Integer(9995), "9995"),
            ("{a:.0G}", NativePrimitive::Integer(9995), "1E+04"),
            ("{a:z.0G}", NativePrimitive::Integer(9995), "1E+04"),
            ("{a:#.0G}", NativePrimitive::Integer(9995), "1.E+04"),
            (
                "{a:+z#014,.2G}",
                NativePrimitive::Integer(9995),
                "+000,001.0E+04",
            ),
            (
                "{a:0=+018_.2G}",
                NativePrimitive::Integer(9995),
                "+0_000_000_001E+04",
            ),
            (
                "{a:*=+018,.2G}",
                NativePrimitive::Integer(9995),
                "+************1E+04",
            ),
            (
                "{a:>018,.2G}",
                NativePrimitive::Integer(9995),
                "00000000000001E+04",
            ),
            (
                "{a:^018_.2G}",
                NativePrimitive::Integer(9995),
                "0000001E+040000000",
            ),
            ("{a:%}", NativePrimitive::Integer(9995), "999500.000000%"),
            ("{a:.0%}", NativePrimitive::Integer(9995), "999500%"),
            ("{a:z.0%}", NativePrimitive::Integer(9995), "999500%"),
            ("{a:#.0%}", NativePrimitive::Integer(9995), "999500.%"),
            (
                "{a:+z#014,.2%}",
                NativePrimitive::Integer(9995),
                "+0,999,500.00%",
            ),
            (
                "{a:0=+018_.2%}",
                NativePrimitive::Integer(9995),
                "+0_000_999_500.00%",
            ),
            (
                "{a:*=+018,.2%}",
                NativePrimitive::Integer(9995),
                "+******999,500.00%",
            ),
            (
                "{a:>018,.2%}",
                NativePrimitive::Integer(9995),
                "0000000999,500.00%",
            ),
            (
                "{a:^018_.2%}",
                NativePrimitive::Integer(9995),
                "000999_500.00%0000",
            ),
        ] {
            let values = BTreeMap::from([("a".to_owned(), primitive(value))]);
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default())?,
                expected,
                "original integer float presentation {template}, {values:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn python_312_integer_float_admission_retains_default_types_and_nested_fields()
    -> Result<(), Box<dyn Error>> {
        for (template, value, expected) in [
            ("{a:d}", NativePrimitive::Integer(0), "0"),
            ("{a:n}", NativePrimitive::Integer(0), "0"),
            ("{a:b}", NativePrimitive::Integer(0), "0"),
            ("{a:o}", NativePrimitive::Integer(0), "0"),
            ("{a:x}", NativePrimitive::Integer(0), "0"),
            ("{a:X}", NativePrimitive::Integer(0), "0"),
            ("{a:}", NativePrimitive::Integer(0), "0"),
            ("{a:d}", NativePrimitive::Integer(7), "7"),
            ("{a:n}", NativePrimitive::Integer(7), "7"),
            ("{a:b}", NativePrimitive::Integer(7), "111"),
            ("{a:o}", NativePrimitive::Integer(7), "7"),
            ("{a:x}", NativePrimitive::Integer(7), "7"),
            ("{a:X}", NativePrimitive::Integer(7), "7"),
            ("{a:}", NativePrimitive::Integer(7), "7"),
            ("{a:d}", NativePrimitive::Integer(-7), "-7"),
            ("{a:n}", NativePrimitive::Integer(-7), "-7"),
            ("{a:b}", NativePrimitive::Integer(-7), "-111"),
            ("{a:o}", NativePrimitive::Integer(-7), "-7"),
            ("{a:x}", NativePrimitive::Integer(-7), "-7"),
            ("{a:X}", NativePrimitive::Integer(-7), "-7"),
            ("{a:}", NativePrimitive::Integer(-7), "-7"),
            (
                "{a:d}",
                NativePrimitive::Integer(9007199254740993),
                "9007199254740993",
            ),
            (
                "{a:n}",
                NativePrimitive::Integer(9007199254740993),
                "9007199254740993",
            ),
            (
                "{a:b}",
                NativePrimitive::Integer(9007199254740993),
                "100000000000000000000000000000000000000000000000000001",
            ),
            (
                "{a:o}",
                NativePrimitive::Integer(9007199254740993),
                "400000000000000001",
            ),
            (
                "{a:x}",
                NativePrimitive::Integer(9007199254740993),
                "20000000000001",
            ),
            (
                "{a:X}",
                NativePrimitive::Integer(9007199254740993),
                "20000000000001",
            ),
            (
                "{a:}",
                NativePrimitive::Integer(9007199254740993),
                "9007199254740993",
            ),
            (
                "{a:d}",
                NativePrimitive::Integer(9007199254740995),
                "9007199254740995",
            ),
            (
                "{a:n}",
                NativePrimitive::Integer(9007199254740995),
                "9007199254740995",
            ),
            (
                "{a:b}",
                NativePrimitive::Integer(9007199254740995),
                "100000000000000000000000000000000000000000000000000011",
            ),
            (
                "{a:o}",
                NativePrimitive::Integer(9007199254740995),
                "400000000000000003",
            ),
            (
                "{a:x}",
                NativePrimitive::Integer(9007199254740995),
                "20000000000003",
            ),
            (
                "{a:X}",
                NativePrimitive::Integer(9007199254740995),
                "20000000000003",
            ),
            (
                "{a:}",
                NativePrimitive::Integer(9007199254740995),
                "9007199254740995",
            ),
            (
                "{a:d}",
                NativePrimitive::Integer(9223372036854775807),
                "9223372036854775807",
            ),
            (
                "{a:n}",
                NativePrimitive::Integer(9223372036854775807),
                "9223372036854775807",
            ),
            (
                "{a:b}",
                NativePrimitive::Integer(9223372036854775807),
                "111111111111111111111111111111111111111111111111111111111111111",
            ),
            (
                "{a:o}",
                NativePrimitive::Integer(9223372036854775807),
                "777777777777777777777",
            ),
            (
                "{a:x}",
                NativePrimitive::Integer(9223372036854775807),
                "7fffffffffffffff",
            ),
            (
                "{a:X}",
                NativePrimitive::Integer(9223372036854775807),
                "7FFFFFFFFFFFFFFF",
            ),
            (
                "{a:}",
                NativePrimitive::Integer(9223372036854775807),
                "9223372036854775807",
            ),
            (
                "{a:d}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9223372036854775808",
            ),
            (
                "{a:n}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9223372036854775808",
            ),
            (
                "{a:b}",
                NativePrimitive::Integer(-9223372036854775808),
                "-1000000000000000000000000000000000000000000000000000000000000000",
            ),
            (
                "{a:o}",
                NativePrimitive::Integer(-9223372036854775808),
                "-1000000000000000000000",
            ),
            (
                "{a:x}",
                NativePrimitive::Integer(-9223372036854775808),
                "-8000000000000000",
            ),
            (
                "{a:X}",
                NativePrimitive::Integer(-9223372036854775808),
                "-8000000000000000",
            ),
            (
                "{a:}",
                NativePrimitive::Integer(-9223372036854775808),
                "-9223372036854775808",
            ),
            (
                "{a:d}",
                NativePrimitive::UnsignedInteger(18446744073709551615),
                "18446744073709551615",
            ),
            (
                "{a:n}",
                NativePrimitive::UnsignedInteger(18446744073709551615),
                "18446744073709551615",
            ),
            (
                "{a:b}",
                NativePrimitive::UnsignedInteger(18446744073709551615),
                "1111111111111111111111111111111111111111111111111111111111111111",
            ),
            (
                "{a:o}",
                NativePrimitive::UnsignedInteger(18446744073709551615),
                "1777777777777777777777",
            ),
            (
                "{a:x}",
                NativePrimitive::UnsignedInteger(18446744073709551615),
                "ffffffffffffffff",
            ),
            (
                "{a:X}",
                NativePrimitive::UnsignedInteger(18446744073709551615),
                "FFFFFFFFFFFFFFFF",
            ),
            (
                "{a:}",
                NativePrimitive::UnsignedInteger(18446744073709551615),
                "18446744073709551615",
            ),
        ] {
            let values = BTreeMap::from([("a".to_owned(), primitive(value))]);
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default())?,
                expected,
                "original integer float presentation {template}, {values:?}"
            );
        }
        let values = BTreeMap::from([
            (
                "a".to_owned(),
                NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value: serde_json::json!({"int": u64::MAX}),
                },
            ),
            (
                "b".to_owned(),
                primitive(NativePrimitive::String("٣０".to_owned())),
            ),
        ]);
        assert_eq!(
            NativeTextFormatter::format(
                "{a[int]:+0{b},.٢f}",
                &values,
                &CancellationToken::default()
            )?,
            "+18,446,744,073,709,551,616.00"
        );
        let values = BTreeMap::from([
            (
                "a".to_owned(),
                primitive(NativePrimitive::Integer(9_007_199_254_740_995)),
            ),
            (
                "b".to_owned(),
                primitive(NativePrimitive::String("٢４".to_owned())),
            ),
            (
                "c".to_owned(),
                primitive(NativePrimitive::String("١７".to_owned())),
            ),
        ]);
        assert_eq!(
            NativeTextFormatter::format(
                "{a:0=+0{b},.{c}g}",
                &values,
                &CancellationToken::default()
            )?,
            "+009,007,199,254,740,996"
        );
        let values = BTreeMap::from([(
            "a".to_owned(),
            NativeValue::PreservedUnknown {
                type_name: "JSON".to_owned(),
                value: serde_json::json!({"bool": false}),
            },
        )]);
        assert_eq!(
            NativeTextFormatter::format("{a[bool]:z#.0f}", &values, &CancellationToken::default())?,
            "0."
        );
        Ok(())
    }

    #[test]
    fn python_312_integer_float_admission_preserves_errors_caps_cancel_and_recovery()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Integer(7)))]);
        for (template, error) in [
            (
                "{missing:.2f}",
                NativeTextFormatError::MissingField("missing".to_owned()),
            ),
            (
                "{a!q:.2f}",
                NativeTextFormatError::InvalidTemplate(
                    "conversion must be exactly !s, !r, or !a".to_owned(),
                ),
            ),
            (
                "{a:0{b}.2f}",
                NativeTextFormatError::MissingField("b".to_owned()),
            ),
            (
                "{a:z.2d}",
                NativeTextFormatError::InvalidTemplate(
                    "integer precision is unsupported".to_owned(),
                ),
            ),
            (
                "{a:z.2q}",
                NativeTextFormatError::InvalidTemplate(
                    "unsupported integer format type".to_owned(),
                ),
            ),
            (
                "{a:0=+016777217,.2q}",
                NativeTextFormatError::InvalidTemplate(
                    "grouping is unsupported for this format type".to_owned(),
                ),
            ),
            (
                "{a:016777217,_.2f}",
                NativeTextFormatError::InvalidTemplate(
                    "cannot combine comma and underscore grouping".to_owned(),
                ),
            ),
            (
                "{a:.f}",
                NativeTextFormatError::InvalidTemplate("precision requires digits".to_owned()),
            ),
            (
                "{a:.2ff}",
                NativeTextFormatError::InvalidTemplate("unsupported format spec".to_owned()),
            ),
        ] {
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default()),
                Err(error),
                "{template}"
            );
        }
        for conversion in ['s', 'r', 'a'] {
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a!{conversion}:.2f}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidTemplate(
                    "numeric format applied to text".to_owned()
                ))
            );
        }
        let overflow = "٩".repeat(40);
        for spec in [format!("0{overflow}.2f"), format!(".{overflow}f")] {
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a:{spec}}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidTemplate(
                    "format number is too large".to_owned()
                ))
            );
        }
        for template in [
            "{a:.16777217f}",
            "{a:.16777217F}",
            "{a:.16777217e}",
            "{a:.16777217E}",
            "{a:.16777217g}",
            "{a:.16777217G}",
            "{a:.16777217%}",
            "{a:016777217,.2f}",
        ] {
            assert!(
                matches!(
                    NativeTextFormatter::format(template, &values, &CancellationToken::default()),
                    Err(NativeTextFormatError::ResultTooLarge { .. })
                ),
                "{template}"
            );
        }
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert_eq!(
            NativeTextFormatter::format("{a:.16777217f}", &values, &cancellation),
            Err(NativeTextFormatError::Cancelled)
        );
        assert_eq!(
            NativeTextFormatter::format("{a:+#.0f}", &values, &CancellationToken::default())?,
            "+7."
        );
        Ok(())
    }

    #[test]
    fn python_312_first_fields_distinguish_all_nd_positions_from_literal_keyword_names()
    -> Result<(), Box<dyn Error>> {
        let mut decimal_count = 0;
        for zero in PYTHON_UNICODE_15_DECIMAL_ZEROES {
            for digit in 0..10 {
                let character =
                    char::from_u32(zero + digit).ok_or("invalid source decimal codepoint")?;
                decimal_count += 1;
                let key = character.to_string();
                let values = BTreeMap::from([
                    (
                        key.clone(),
                        primitive(NativePrimitive::String("must-not-alias-kwargs".to_owned())),
                    ),
                    (
                        "a".to_owned(),
                        primitive(NativePrimitive::String("x".to_owned())),
                    ),
                ]);
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{{key}}}"),
                        &values,
                        &CancellationToken::default()
                    ),
                    Err(NativeTextFormatError::InvalidFieldAccess(key.clone()))
                );
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{a:>{{{key}}}}}"),
                        &values,
                        &CancellationToken::default()
                    ),
                    Err(NativeTextFormatError::InvalidFieldAccess(key.clone()))
                );
                let key = format!("{key}a");
                let values = BTreeMap::from([(
                    key.clone(),
                    primitive(NativePrimitive::String("literal-key".to_owned())),
                )]);
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{{key}}}"),
                        &values,
                        &CancellationToken::default()
                    )?,
                    "literal-key"
                );
                let values = BTreeMap::from([
                    (
                        key.clone(),
                        primitive(NativePrimitive::String("5".to_owned())),
                    ),
                    (
                        "a".to_owned(),
                        primitive(NativePrimitive::String("x".to_owned())),
                    ),
                ]);
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{a:>{{{key}}}}}"),
                        &values,
                        &CancellationToken::default()
                    )?,
                    "    x"
                );
            }
        }
        assert_eq!(decimal_count, 680);
        Ok(())
    }

    #[test]
    fn python_312_first_fields_preserve_all_nondecimal_numeric_literal_keys()
    -> Result<(), Box<dyn Error>> {
        let source_codepoints = [
            0xb2, 0xb3, 0xb9, 0xbc, 0xbd, 0xbe, 0x9f4, 0x9f5, 0x9f6, 0x9f7, 0x9f8, 0x9f9, 0xb72,
            0xb73, 0xb74, 0xb75, 0xb76, 0xb77, 0xbf0, 0xbf1, 0xbf2, 0xc78, 0xc79, 0xc7a, 0xc7b,
            0xc7c, 0xc7d, 0xc7e, 0xd58, 0xd59, 0xd5a, 0xd5b, 0xd5c, 0xd5d, 0xd5e, 0xd70, 0xd71,
            0xd72, 0xd73, 0xd74, 0xd75, 0xd76, 0xd77, 0xd78, 0xf2a, 0xf2b, 0xf2c, 0xf2d, 0xf2e,
            0xf2f, 0xf30, 0xf31, 0xf32, 0xf33, 0x1369, 0x136a, 0x136b, 0x136c, 0x136d, 0x136e,
            0x136f, 0x1370, 0x1371, 0x1372, 0x1373, 0x1374, 0x1375, 0x1376, 0x1377, 0x1378, 0x1379,
            0x137a, 0x137b, 0x137c, 0x16ee, 0x16ef, 0x16f0, 0x17f0, 0x17f1, 0x17f2, 0x17f3, 0x17f4,
            0x17f5, 0x17f6, 0x17f7, 0x17f8, 0x17f9, 0x19da, 0x2070, 0x2074, 0x2075, 0x2076, 0x2077,
            0x2078, 0x2079, 0x2080, 0x2081, 0x2082, 0x2083, 0x2084, 0x2085, 0x2086, 0x2087, 0x2088,
            0x2089, 0x2150, 0x2151, 0x2152, 0x2153, 0x2154, 0x2155, 0x2156, 0x2157, 0x2158, 0x2159,
            0x215a, 0x215b, 0x215c, 0x215d, 0x215e, 0x215f, 0x2160, 0x2161, 0x2162, 0x2163, 0x2164,
            0x2165, 0x2166, 0x2167, 0x2168, 0x2169, 0x216a, 0x216b, 0x216c, 0x216d, 0x216e, 0x216f,
            0x2170, 0x2171, 0x2172, 0x2173, 0x2174, 0x2175, 0x2176, 0x2177, 0x2178, 0x2179, 0x217a,
            0x217b, 0x217c, 0x217d, 0x217e, 0x217f, 0x2180, 0x2181, 0x2182, 0x2185, 0x2186, 0x2187,
            0x2188, 0x2189, 0x2460, 0x2461, 0x2462, 0x2463, 0x2464, 0x2465, 0x2466, 0x2467, 0x2468,
            0x2469, 0x246a, 0x246b, 0x246c, 0x246d, 0x246e, 0x246f, 0x2470, 0x2471, 0x2472, 0x2473,
            0x2474, 0x2475, 0x2476, 0x2477, 0x2478, 0x2479, 0x247a, 0x247b, 0x247c, 0x247d, 0x247e,
            0x247f, 0x2480, 0x2481, 0x2482, 0x2483, 0x2484, 0x2485, 0x2486, 0x2487, 0x2488, 0x2489,
            0x248a, 0x248b, 0x248c, 0x248d, 0x248e, 0x248f, 0x2490, 0x2491, 0x2492, 0x2493, 0x2494,
            0x2495, 0x2496, 0x2497, 0x2498, 0x2499, 0x249a, 0x249b, 0x24ea, 0x24eb, 0x24ec, 0x24ed,
            0x24ee, 0x24ef, 0x24f0, 0x24f1, 0x24f2, 0x24f3, 0x24f4, 0x24f5, 0x24f6, 0x24f7, 0x24f8,
            0x24f9, 0x24fa, 0x24fb, 0x24fc, 0x24fd, 0x24fe, 0x24ff, 0x2776, 0x2777, 0x2778, 0x2779,
            0x277a, 0x277b, 0x277c, 0x277d, 0x277e, 0x277f, 0x2780, 0x2781, 0x2782, 0x2783, 0x2784,
            0x2785, 0x2786, 0x2787, 0x2788, 0x2789, 0x278a, 0x278b, 0x278c, 0x278d, 0x278e, 0x278f,
            0x2790, 0x2791, 0x2792, 0x2793, 0x2cfd, 0x3007, 0x3021, 0x3022, 0x3023, 0x3024, 0x3025,
            0x3026, 0x3027, 0x3028, 0x3029, 0x3038, 0x3039, 0x303a, 0x3192, 0x3193, 0x3194, 0x3195,
            0x3220, 0x3221, 0x3222, 0x3223, 0x3224, 0x3225, 0x3226, 0x3227, 0x3228, 0x3229, 0x3248,
            0x3249, 0x324a, 0x324b, 0x324c, 0x324d, 0x324e, 0x324f, 0x3251, 0x3252, 0x3253, 0x3254,
            0x3255, 0x3256, 0x3257, 0x3258, 0x3259, 0x325a, 0x325b, 0x325c, 0x325d, 0x325e, 0x325f,
            0x3280, 0x3281, 0x3282, 0x3283, 0x3284, 0x3285, 0x3286, 0x3287, 0x3288, 0x3289, 0x32b1,
            0x32b2, 0x32b3, 0x32b4, 0x32b5, 0x32b6, 0x32b7, 0x32b8, 0x32b9, 0x32ba, 0x32bb, 0x32bc,
            0x32bd, 0x32be, 0x32bf, 0x3405, 0x3483, 0x382a, 0x3b4d, 0x4e00, 0x4e03, 0x4e07, 0x4e09,
            0x4e5d, 0x4e8c, 0x4e94, 0x4e96, 0x4ebf, 0x4ec0, 0x4edf, 0x4ee8, 0x4f0d, 0x4f70, 0x5104,
            0x5146, 0x5169, 0x516b, 0x516d, 0x5341, 0x5343, 0x5344, 0x5345, 0x534c, 0x53c1, 0x53c2,
            0x53c3, 0x53c4, 0x56db, 0x58f1, 0x58f9, 0x5e7a, 0x5efe, 0x5eff, 0x5f0c, 0x5f0d, 0x5f0e,
            0x5f10, 0x62fe, 0x634c, 0x67d2, 0x6f06, 0x7396, 0x767e, 0x8086, 0x842c, 0x8cae, 0x8cb3,
            0x8d30, 0x9621, 0x9646, 0x964c, 0x9678, 0x96f6, 0xa6e6, 0xa6e7, 0xa6e8, 0xa6e9, 0xa6ea,
            0xa6eb, 0xa6ec, 0xa6ed, 0xa6ee, 0xa6ef, 0xa830, 0xa831, 0xa832, 0xa833, 0xa834, 0xa835,
            0xf96b, 0xf973, 0xf978, 0xf9b2, 0xf9d1, 0xf9d3, 0xf9fd, 0x10107, 0x10108, 0x10109,
            0x1010a, 0x1010b, 0x1010c, 0x1010d, 0x1010e, 0x1010f, 0x10110, 0x10111, 0x10112,
            0x10113, 0x10114, 0x10115, 0x10116, 0x10117, 0x10118, 0x10119, 0x1011a, 0x1011b,
            0x1011c, 0x1011d, 0x1011e, 0x1011f, 0x10120, 0x10121, 0x10122, 0x10123, 0x10124,
            0x10125, 0x10126, 0x10127, 0x10128, 0x10129, 0x1012a, 0x1012b, 0x1012c, 0x1012d,
            0x1012e, 0x1012f, 0x10130, 0x10131, 0x10132, 0x10133, 0x10140, 0x10141, 0x10142,
            0x10143, 0x10144, 0x10145, 0x10146, 0x10147, 0x10148, 0x10149, 0x1014a, 0x1014b,
            0x1014c, 0x1014d, 0x1014e, 0x1014f, 0x10150, 0x10151, 0x10152, 0x10153, 0x10154,
            0x10155, 0x10156, 0x10157, 0x10158, 0x10159, 0x1015a, 0x1015b, 0x1015c, 0x1015d,
            0x1015e, 0x1015f, 0x10160, 0x10161, 0x10162, 0x10163, 0x10164, 0x10165, 0x10166,
            0x10167, 0x10168, 0x10169, 0x1016a, 0x1016b, 0x1016c, 0x1016d, 0x1016e, 0x1016f,
            0x10170, 0x10171, 0x10172, 0x10173, 0x10174, 0x10175, 0x10176, 0x10177, 0x10178,
            0x1018a, 0x1018b, 0x102e1, 0x102e2, 0x102e3, 0x102e4, 0x102e5, 0x102e6, 0x102e7,
            0x102e8, 0x102e9, 0x102ea, 0x102eb, 0x102ec, 0x102ed, 0x102ee, 0x102ef, 0x102f0,
            0x102f1, 0x102f2, 0x102f3, 0x102f4, 0x102f5, 0x102f6, 0x102f7, 0x102f8, 0x102f9,
            0x102fa, 0x102fb, 0x10320, 0x10321, 0x10322, 0x10323, 0x10341, 0x1034a, 0x103d1,
            0x103d2, 0x103d3, 0x103d4, 0x103d5, 0x10858, 0x10859, 0x1085a, 0x1085b, 0x1085c,
            0x1085d, 0x1085e, 0x1085f, 0x10879, 0x1087a, 0x1087b, 0x1087c, 0x1087d, 0x1087e,
            0x1087f, 0x108a7, 0x108a8, 0x108a9, 0x108aa, 0x108ab, 0x108ac, 0x108ad, 0x108ae,
            0x108af, 0x108fb, 0x108fc, 0x108fd, 0x108fe, 0x108ff, 0x10916, 0x10917, 0x10918,
            0x10919, 0x1091a, 0x1091b, 0x109bc, 0x109bd, 0x109c0, 0x109c1, 0x109c2, 0x109c3,
            0x109c4, 0x109c5, 0x109c6, 0x109c7, 0x109c8, 0x109c9, 0x109ca, 0x109cb, 0x109cc,
            0x109cd, 0x109ce, 0x109cf, 0x109d2, 0x109d3, 0x109d4, 0x109d5, 0x109d6, 0x109d7,
            0x109d8, 0x109d9, 0x109da, 0x109db, 0x109dc, 0x109dd, 0x109de, 0x109df, 0x109e0,
            0x109e1, 0x109e2, 0x109e3, 0x109e4, 0x109e5, 0x109e6, 0x109e7, 0x109e8, 0x109e9,
            0x109ea, 0x109eb, 0x109ec, 0x109ed, 0x109ee, 0x109ef, 0x109f0, 0x109f1, 0x109f2,
            0x109f3, 0x109f4, 0x109f5, 0x109f6, 0x109f7, 0x109f8, 0x109f9, 0x109fa, 0x109fb,
            0x109fc, 0x109fd, 0x109fe, 0x109ff, 0x10a40, 0x10a41, 0x10a42, 0x10a43, 0x10a44,
            0x10a45, 0x10a46, 0x10a47, 0x10a48, 0x10a7d, 0x10a7e, 0x10a9d, 0x10a9e, 0x10a9f,
            0x10aeb, 0x10aec, 0x10aed, 0x10aee, 0x10aef, 0x10b58, 0x10b59, 0x10b5a, 0x10b5b,
            0x10b5c, 0x10b5d, 0x10b5e, 0x10b5f, 0x10b78, 0x10b79, 0x10b7a, 0x10b7b, 0x10b7c,
            0x10b7d, 0x10b7e, 0x10b7f, 0x10ba9, 0x10baa, 0x10bab, 0x10bac, 0x10bad, 0x10bae,
            0x10baf, 0x10cfa, 0x10cfb, 0x10cfc, 0x10cfd, 0x10cfe, 0x10cff, 0x10e60, 0x10e61,
            0x10e62, 0x10e63, 0x10e64, 0x10e65, 0x10e66, 0x10e67, 0x10e68, 0x10e69, 0x10e6a,
            0x10e6b, 0x10e6c, 0x10e6d, 0x10e6e, 0x10e6f, 0x10e70, 0x10e71, 0x10e72, 0x10e73,
            0x10e74, 0x10e75, 0x10e76, 0x10e77, 0x10e78, 0x10e79, 0x10e7a, 0x10e7b, 0x10e7c,
            0x10e7d, 0x10e7e, 0x10f1d, 0x10f1e, 0x10f1f, 0x10f20, 0x10f21, 0x10f22, 0x10f23,
            0x10f24, 0x10f25, 0x10f26, 0x10f51, 0x10f52, 0x10f53, 0x10f54, 0x10fc5, 0x10fc6,
            0x10fc7, 0x10fc8, 0x10fc9, 0x10fca, 0x10fcb, 0x11052, 0x11053, 0x11054, 0x11055,
            0x11056, 0x11057, 0x11058, 0x11059, 0x1105a, 0x1105b, 0x1105c, 0x1105d, 0x1105e,
            0x1105f, 0x11060, 0x11061, 0x11062, 0x11063, 0x11064, 0x11065, 0x111e1, 0x111e2,
            0x111e3, 0x111e4, 0x111e5, 0x111e6, 0x111e7, 0x111e8, 0x111e9, 0x111ea, 0x111eb,
            0x111ec, 0x111ed, 0x111ee, 0x111ef, 0x111f0, 0x111f1, 0x111f2, 0x111f3, 0x111f4,
            0x1173a, 0x1173b, 0x118ea, 0x118eb, 0x118ec, 0x118ed, 0x118ee, 0x118ef, 0x118f0,
            0x118f1, 0x118f2, 0x11c5a, 0x11c5b, 0x11c5c, 0x11c5d, 0x11c5e, 0x11c5f, 0x11c60,
            0x11c61, 0x11c62, 0x11c63, 0x11c64, 0x11c65, 0x11c66, 0x11c67, 0x11c68, 0x11c69,
            0x11c6a, 0x11c6b, 0x11c6c, 0x11fc0, 0x11fc1, 0x11fc2, 0x11fc3, 0x11fc4, 0x11fc5,
            0x11fc6, 0x11fc7, 0x11fc8, 0x11fc9, 0x11fca, 0x11fcb, 0x11fcc, 0x11fcd, 0x11fce,
            0x11fcf, 0x11fd0, 0x11fd1, 0x11fd2, 0x11fd3, 0x11fd4, 0x12400, 0x12401, 0x12402,
            0x12403, 0x12404, 0x12405, 0x12406, 0x12407, 0x12408, 0x12409, 0x1240a, 0x1240b,
            0x1240c, 0x1240d, 0x1240e, 0x1240f, 0x12410, 0x12411, 0x12412, 0x12413, 0x12414,
            0x12415, 0x12416, 0x12417, 0x12418, 0x12419, 0x1241a, 0x1241b, 0x1241c, 0x1241d,
            0x1241e, 0x1241f, 0x12420, 0x12421, 0x12422, 0x12423, 0x12424, 0x12425, 0x12426,
            0x12427, 0x12428, 0x12429, 0x1242a, 0x1242b, 0x1242c, 0x1242d, 0x1242e, 0x1242f,
            0x12430, 0x12431, 0x12432, 0x12433, 0x12434, 0x12435, 0x12436, 0x12437, 0x12438,
            0x12439, 0x1243a, 0x1243b, 0x1243c, 0x1243d, 0x1243e, 0x1243f, 0x12440, 0x12441,
            0x12442, 0x12443, 0x12444, 0x12445, 0x12446, 0x12447, 0x12448, 0x12449, 0x1244a,
            0x1244b, 0x1244c, 0x1244d, 0x1244e, 0x1244f, 0x12450, 0x12451, 0x12452, 0x12453,
            0x12454, 0x12455, 0x12456, 0x12457, 0x12458, 0x12459, 0x1245a, 0x1245b, 0x1245c,
            0x1245d, 0x1245e, 0x1245f, 0x12460, 0x12461, 0x12462, 0x12463, 0x12464, 0x12465,
            0x12466, 0x12467, 0x12468, 0x12469, 0x1246a, 0x1246b, 0x1246c, 0x1246d, 0x1246e,
            0x16b5b, 0x16b5c, 0x16b5d, 0x16b5e, 0x16b5f, 0x16b60, 0x16b61, 0x16e80, 0x16e81,
            0x16e82, 0x16e83, 0x16e84, 0x16e85, 0x16e86, 0x16e87, 0x16e88, 0x16e89, 0x16e8a,
            0x16e8b, 0x16e8c, 0x16e8d, 0x16e8e, 0x16e8f, 0x16e90, 0x16e91, 0x16e92, 0x16e93,
            0x16e94, 0x16e95, 0x16e96, 0x1d2c0, 0x1d2c1, 0x1d2c2, 0x1d2c3, 0x1d2c4, 0x1d2c5,
            0x1d2c6, 0x1d2c7, 0x1d2c8, 0x1d2c9, 0x1d2ca, 0x1d2cb, 0x1d2cc, 0x1d2cd, 0x1d2ce,
            0x1d2cf, 0x1d2d0, 0x1d2d1, 0x1d2d2, 0x1d2d3, 0x1d2e0, 0x1d2e1, 0x1d2e2, 0x1d2e3,
            0x1d2e4, 0x1d2e5, 0x1d2e6, 0x1d2e7, 0x1d2e8, 0x1d2e9, 0x1d2ea, 0x1d2eb, 0x1d2ec,
            0x1d2ed, 0x1d2ee, 0x1d2ef, 0x1d2f0, 0x1d2f1, 0x1d2f2, 0x1d2f3, 0x1d360, 0x1d361,
            0x1d362, 0x1d363, 0x1d364, 0x1d365, 0x1d366, 0x1d367, 0x1d368, 0x1d369, 0x1d36a,
            0x1d36b, 0x1d36c, 0x1d36d, 0x1d36e, 0x1d36f, 0x1d370, 0x1d371, 0x1d372, 0x1d373,
            0x1d374, 0x1d375, 0x1d376, 0x1d377, 0x1d378, 0x1e8c7, 0x1e8c8, 0x1e8c9, 0x1e8ca,
            0x1e8cb, 0x1e8cc, 0x1e8cd, 0x1e8ce, 0x1e8cf, 0x1ec71, 0x1ec72, 0x1ec73, 0x1ec74,
            0x1ec75, 0x1ec76, 0x1ec77, 0x1ec78, 0x1ec79, 0x1ec7a, 0x1ec7b, 0x1ec7c, 0x1ec7d,
            0x1ec7e, 0x1ec7f, 0x1ec80, 0x1ec81, 0x1ec82, 0x1ec83, 0x1ec84, 0x1ec85, 0x1ec86,
            0x1ec87, 0x1ec88, 0x1ec89, 0x1ec8a, 0x1ec8b, 0x1ec8c, 0x1ec8d, 0x1ec8e, 0x1ec8f,
            0x1ec90, 0x1ec91, 0x1ec92, 0x1ec93, 0x1ec94, 0x1ec95, 0x1ec96, 0x1ec97, 0x1ec98,
            0x1ec99, 0x1ec9a, 0x1ec9b, 0x1ec9c, 0x1ec9d, 0x1ec9e, 0x1ec9f, 0x1eca0, 0x1eca1,
            0x1eca2, 0x1eca3, 0x1eca4, 0x1eca5, 0x1eca6, 0x1eca7, 0x1eca8, 0x1eca9, 0x1ecaa,
            0x1ecab, 0x1ecad, 0x1ecae, 0x1ecaf, 0x1ecb1, 0x1ecb2, 0x1ecb3, 0x1ecb4, 0x1ed01,
            0x1ed02, 0x1ed03, 0x1ed04, 0x1ed05, 0x1ed06, 0x1ed07, 0x1ed08, 0x1ed09, 0x1ed0a,
            0x1ed0b, 0x1ed0c, 0x1ed0d, 0x1ed0e, 0x1ed0f, 0x1ed10, 0x1ed11, 0x1ed12, 0x1ed13,
            0x1ed14, 0x1ed15, 0x1ed16, 0x1ed17, 0x1ed18, 0x1ed19, 0x1ed1a, 0x1ed1b, 0x1ed1c,
            0x1ed1d, 0x1ed1e, 0x1ed1f, 0x1ed20, 0x1ed21, 0x1ed22, 0x1ed23, 0x1ed24, 0x1ed25,
            0x1ed26, 0x1ed27, 0x1ed28, 0x1ed29, 0x1ed2a, 0x1ed2b, 0x1ed2c, 0x1ed2d, 0x1ed2f,
            0x1ed30, 0x1ed31, 0x1ed32, 0x1ed33, 0x1ed34, 0x1ed35, 0x1ed36, 0x1ed37, 0x1ed38,
            0x1ed39, 0x1ed3a, 0x1ed3b, 0x1ed3c, 0x1ed3d, 0x1f100, 0x1f101, 0x1f102, 0x1f103,
            0x1f104, 0x1f105, 0x1f106, 0x1f107, 0x1f108, 0x1f109, 0x1f10a, 0x1f10b, 0x1f10c,
            0x20001, 0x20064, 0x200e2, 0x20121, 0x2092a, 0x20983, 0x2098c, 0x2099c, 0x20aea,
            0x20afd, 0x20b19, 0x22390, 0x22998, 0x23b1b, 0x2626d, 0x2f890,
        ];
        assert_eq!(source_codepoints.len(), 1232);
        for codepoint in source_codepoints {
            let character =
                char::from_u32(codepoint).ok_or("invalid source nondecimal codepoint")?;
            let key = character.to_string();
            let values = BTreeMap::from([(
                key.clone(),
                primitive(NativePrimitive::String("literal-key".to_owned())),
            )]);
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{{key}}}"),
                    &values,
                    &CancellationToken::default()
                )?,
                "literal-key"
            );
            let values = BTreeMap::from([
                (
                    key.clone(),
                    primitive(NativePrimitive::String("5".to_owned())),
                ),
                (
                    "a".to_owned(),
                    primitive(NativePrimitive::String("x".to_owned())),
                ),
            ]);
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a:>{{{key}}}}}"),
                    &values,
                    &CancellationToken::default()
                )?,
                "    x"
            );
        }
        Ok(())
    }

    #[test]
    fn python_312_first_field_decimal_overflow_precedes_trailing_text_or_kwargs_lookup()
    -> Result<(), Box<dyn Error>> {
        for key in [
            "0".to_owned(),
            "01".to_owned(),
            "٠".to_owned(),
            "٠١".to_owned(),
            "١４".to_owned(),
            "00١０".to_owned(),
            "0".repeat(80) + "1",
            "٠".repeat(80) + "1",
        ] {
            let values = BTreeMap::from([(
                key.clone(),
                primitive(NativePrimitive::String("must-not-alias".to_owned())),
            )]);
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{{key}}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidFieldAccess(key))
            );
        }
        for key in [
            "9".repeat(40) + "a",
            "٩".repeat(40) + "a",
            "9".repeat(40) + "¹",
            "٠".repeat(80) + &"٩".repeat(40) + "a",
            (isize::MAX as u128 + 1).to_string() + "a",
        ] {
            let values = BTreeMap::from([(
                key.clone(),
                primitive(NativePrimitive::String(
                    "must-not-bypass-overflow".to_owned(),
                )),
            )]);
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{{key}}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidFieldAccess(key))
            );
        }
        for key in [
            "+1".to_owned(),
            "-1".to_owned(),
            " 1".to_owned(),
            "1 ".to_owned(),
            "1a".to_owned(),
            "١a".to_owned(),
            "1١a".to_owned(),
            "¹a".to_owned(),
            "1²".to_owned(),
            "0x10".to_owned(),
            "😀".to_owned(),
            "a-b".to_owned(),
            "a b".to_owned(),
            "__import__".to_owned(),
            "\u{10d40}".to_owned(),
            "0".repeat(80) + "x",
            "٠".repeat(80) + "x",
            "0".repeat(80) + "1a",
            "+".to_owned() + &"9".repeat(40) + "a",
            "1²".to_owned() + &"9".repeat(40),
            isize::MAX.to_string() + "a",
        ] {
            let values = BTreeMap::from([(
                key.clone(),
                primitive(NativePrimitive::String("literal-key".to_owned())),
            )]);
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{{key}}}"),
                    &values,
                    &CancellationToken::default()
                )?,
                "literal-key"
            );
        }
        Ok(())
    }

    #[test]
    fn python_312_first_field_lookup_retains_nested_access_error_boundaries_and_cancel()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([
            ("١a".to_owned(), primitive(NativePrimitive::Integer(7))),
            ("a".to_owned(), primitive(NativePrimitive::Integer(7))),
        ]);
        assert_eq!(
            NativeTextFormatter::format("{١a.real}", &values, &CancellationToken::default())?,
            "7"
        );
        let values = BTreeMap::from([(
            "١a".to_owned(),
            NativeValue::List {
                values: vec![primitive(NativePrimitive::String("literal-key".to_owned()))],
            },
        )]);
        assert_eq!(
            NativeTextFormatter::format("{١a[0]}", &values, &CancellationToken::default())?,
            "literal-key"
        );
        let values = BTreeMap::from([
            ("a".to_owned(), primitive(NativePrimitive::Integer(7))),
            (
                "١a".to_owned(),
                primitive(NativePrimitive::String("5".to_owned())),
            ),
        ]);
        assert_eq!(
            NativeTextFormatter::format("{a:0{١a}d}", &values, &CancellationToken::default())?,
            "00007"
        );
        let values = BTreeMap::from([
            ("a".to_owned(), primitive(NativePrimitive::Integer(7))),
            (
                "١a".to_owned(),
                primitive(NativePrimitive::String("2".to_owned())),
            ),
        ]);
        assert_eq!(
            NativeTextFormatter::format("{a:.{١a}f}", &values, &CancellationToken::default())?,
            "7.00"
        );
        for (template, error) in [
            (
                "{missing:>{١}}",
                NativeTextFormatError::MissingField("missing".to_owned()),
            ),
            (
                "{١:>{missing}}",
                NativeTextFormatError::InvalidFieldAccess("١".to_owned()),
            ),
            (
                "{١!s}",
                NativeTextFormatError::InvalidFieldAccess("١".to_owned()),
            ),
            (
                "{١.real}",
                NativeTextFormatError::InvalidFieldAccess("١.real".to_owned()),
            ),
            (
                "{١[0]}",
                NativeTextFormatError::InvalidFieldAccess("١[0]".to_owned()),
            ),
            ("{١a}", NativeTextFormatError::MissingField("١a".to_owned())),
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    template,
                    &BTreeMap::new(),
                    &CancellationToken::default()
                ),
                Err(error)
            );
        }
        for template in ["{}", "{}{0}", "{a}{}", "{a:>{}}"] {
            assert!(
                matches!(NativeTextFormatter::format(template, &values, &CancellationToken::default()), Err(NativeTextFormatError::InvalidTemplate(error)) if error == "keyword-only formatting requires a field name")
            );
        }
        assert_eq!(
            NativeTextFormatter::format("{0}{}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::InvalidFieldAccess("0".to_owned()))
        );
        assert_eq!(
            NativeTextFormatter::format("{{١}}", &BTreeMap::new(), &CancellationToken::default())?,
            "{١}"
        );
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert_eq!(
            NativeTextFormatter::format("{١a}", &BTreeMap::new(), &cancellation),
            Err(NativeTextFormatError::Cancelled)
        );
        assert_eq!(
            NativeTextFormatter::format(
                &"a".repeat(NATIVE_TEXT_FORMAT_MAX_TEMPLATE_BYTES + 1),
                &values,
                &CancellationToken::default()
            ),
            Err(NativeTextFormatError::TemplateTooLarge {
                actual_bytes: NATIVE_TEXT_FORMAT_MAX_TEMPLATE_BYTES + 1,
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_TEMPLATE_BYTES
            })
        );
        Ok(())
    }

    #[test]
    fn python_312_unknown_conversions_follow_first_field_and_access_lookup()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([
            (
                "a".to_owned(),
                NativeValue::PreservedUnknown {
                    type_name: "record".to_owned(),
                    value: serde_json::json!({"present": 7, "!q": "literal", ":": "colon"}),
                },
            ),
            (
                "١".to_owned(),
                primitive(NativePrimitive::String("must-not-alias".to_owned())),
            ),
        ]);
        for conversion in ['q', 'x', '!', '}', ':', '{', '0', '١', '😀', ' ', '\u{1c}'] {
            for (field, error) in [
                (
                    "missing".to_owned(),
                    NativeTextFormatError::MissingField("missing".to_owned()),
                ),
                (
                    "missing[0]".to_owned(),
                    NativeTextFormatError::MissingField("missing".to_owned()),
                ),
                (
                    "0".to_owned(),
                    NativeTextFormatError::InvalidFieldAccess("0".to_owned()),
                ),
                (
                    "١".to_owned(),
                    NativeTextFormatError::InvalidFieldAccess("١".to_owned()),
                ),
                (
                    "٩".repeat(40) + "a",
                    NativeTextFormatError::InvalidFieldAccess("٩".repeat(40) + "a"),
                ),
                (
                    "a[missing]".to_owned(),
                    NativeTextFormatError::InvalidFieldAccess("a[missing]".to_owned()),
                ),
                (
                    "a.unknown".to_owned(),
                    NativeTextFormatError::InvalidFieldAccess("a.unknown".to_owned()),
                ),
            ] {
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{{field}!{conversion}}}"),
                        &values,
                        &CancellationToken::default()
                    ),
                    Err(error)
                );
            }
            for field in ["a", "a[present]", "a[!q]", "a[:]"] {
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{{field}!{conversion}}}"),
                        &values,
                        &CancellationToken::default()
                    ),
                    Err(invalid_conversion())
                );
            }
        }
        Ok(())
    }

    #[test]
    fn python_312_conversion_syntax_errors_precede_missing_or_positional_lookup() {
        let values = BTreeMap::new();
        for field in ["missing", "0", "١", "a[missing]"] {
            for conversion in ["ss", "qq", "sq", "s!q", "q!s", ":s", "s[0]"] {
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{{field}!{conversion}}}"),
                        &values,
                        &CancellationToken::default()
                    ),
                    Err(invalid_conversion())
                );
            }
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{{field}!}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidTemplate(
                    "replacement field is unterminated".to_owned()
                ))
            );
        }
        for template in [
            "{missing!q:{{}",
            "{missing!q:>{width}",
            "{missing!q:broken{",
        ] {
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default()),
                Err(NativeTextFormatError::InvalidTemplate(
                    "replacement field is unterminated".to_owned()
                ))
            );
        }
    }

    #[test]
    fn python_312_unknown_conversion_precedes_nested_field_and_spec_admission()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([
            ("a".to_owned(), primitive(NativePrimitive::Integer(7))),
            ("width".to_owned(), primitive(NativePrimitive::Integer(5))),
        ]);
        for template in [
            "{a!q:{missing_width}}",
            "{a!q:{missing_width!qq}}",
            "{a!q:{width!q}}",
            "{a!q:{width!ss}}",
            "{a!q:{{}}}",
            "{a!q:.2q}",
            "{a!q}}",
            "{a!q} {missing!qq}",
        ] {
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default()),
                Err(invalid_conversion())
            );
        }
        for template in [
            "{missing!q:{width!ss}}",
            "{missing!q:{width!q}}",
            "{missing!q:.2q}",
        ] {
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default()),
                Err(NativeTextFormatError::MissingField("missing".to_owned()))
            );
        }
        assert_eq!(
            NativeTextFormatter::format(
                "{a!s:{missing_width!q}}",
                &values,
                &CancellationToken::default()
            ),
            Err(NativeTextFormatError::MissingField(
                "missing_width".to_owned()
            ))
        );
        assert_eq!(
            NativeTextFormatter::format("{a!s:{width!q}}", &values, &CancellationToken::default()),
            Err(invalid_conversion())
        );
        assert_eq!(
            NativeTextFormatter::format("{a!s:>{width}}", &values, &CancellationToken::default())?,
            "    7"
        );
        Ok(())
    }

    #[test]
    fn python_312_conversion_delimiters_preserve_literal_dictionary_and_nested_values()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([
            (
                "a".to_owned(),
                NativeValue::PreservedUnknown {
                    type_name: "record".to_owned(),
                    value: serde_json::json!({"!q": "literal", "missing!q": "both", ":": "colon"}),
                },
            ),
            ("width".to_owned(), primitive(NativePrimitive::Integer(9))),
            (
                "word".to_owned(),
                primitive(NativePrimitive::String("é".to_owned())),
            ),
        ]);
        for (template, expected) in [
            ("{a[!q]!s}", "literal"),
            ("{a[missing!q]!r}", "'both'"),
            ("{a[:]!s}", "colon"),
            ("{word!s:>{width}}", "        é"),
            ("{word!r:>{width}}", "      'é'"),
            ("{word!a:>{width}}", "   '\\xe9'"),
        ] {
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default())?,
                expected
            );
        }
        assert_eq!(
            NativeTextFormatter::format(
                "{{a!q}} {a[!q]!s}",
                &values,
                &CancellationToken::default()
            )?,
            "{a!q} literal"
        );
        Ok(())
    }

    #[test]
    fn python_312_conversion_phase_correction_retains_caps_cancel_and_recovery()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Integer(7)))]);
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        assert_eq!(
            NativeTextFormatter::format("{missing!qq}", &values, &cancellation),
            Err(NativeTextFormatError::Cancelled)
        );
        let oversized = "x".repeat(NATIVE_TEXT_FORMAT_MAX_TEMPLATE_BYTES + 1);
        assert_eq!(
            NativeTextFormatter::format(&oversized, &values, &CancellationToken::default()),
            Err(NativeTextFormatError::TemplateTooLarge {
                actual_bytes: oversized.len(),
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_TEMPLATE_BYTES
            })
        );
        assert_eq!(
            NativeTextFormatter::format("{a!q}", &values, &CancellationToken::default()),
            Err(invalid_conversion())
        );
        assert_eq!(
            NativeTextFormatter::format("{a!s:>{missing}}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::MissingField("missing".to_owned()))
        );
        assert_eq!(
            NativeTextFormatter::format("{a!s:05s}", &values, &CancellationToken::default())?,
            "70000"
        );
        Ok(())
    }

    fn unicode_source_fixture() -> Result<Value, Box<dyn Error>> {
        Ok(serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sim_test_support/fixtures/nodes/text-format-python-312/source-contract.json"
        )))?)
    }

    #[test]
    fn pinned_python_finite_binary64_conversions_are_bit_exact() -> Result<(), Box<dyn Error>> {
        let fixture = unicode_source_fixture()?;
        let cases = fixture["finite_float_cases"]
            .as_array()
            .ok_or("missing finite float cases")?;
        assert_eq!(cases.len(), 2171);
        for case in cases {
            let bits =
                u64::from_str_radix(case["bits_hex"].as_str().ok_or("missing float bits")?, 16)?;
            let value = f64::from_bits(bits);
            assert!(value.is_finite());
            let values =
                BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Number(value)))]);
            assert_eq!(
                NativeTextFormatter::format(
                    case["template"].as_str().ok_or("missing template")?,
                    &values,
                    &CancellationToken::default()
                )?,
                case["output"].as_str().ok_or("missing original output")?,
                "source binary64 case {}",
                case["id"]
            );
        }
        Ok(())
    }

    #[test]
    fn pinned_python_float_container_repr_preserves_integer_types() -> Result<(), Box<dyn Error>> {
        let fixture = unicode_source_fixture()?;
        let cases = fixture["finite_float_container_cases"]
            .as_array()
            .ok_or("missing float containers")?;
        assert_eq!(cases.len(), 4);
        for case in cases {
            let bits = u64::from_str_radix(
                case["float_bits_hex"]
                    .as_str()
                    .ok_or("missing float bits")?,
                16,
            )?;
            let number = serde_json::Number::from_f64(f64::from_bits(bits))
                .ok_or("nonfinite container value")?;
            let value =
                serde_json::json!({"a": [number.clone(), 7, -3, u64::MAX], "b": {"c": number}});
            let values = BTreeMap::from([(
                "a".to_owned(),
                NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value,
                },
            )]);
            assert_eq!(
                NativeTextFormatter::format(
                    case["template"].as_str().ok_or("missing template")?,
                    &values,
                    &CancellationToken::default()
                )?,
                case["output"].as_str().ok_or("missing original output")?,
                "source float container {}",
                case["id"]
            );
        }
        Ok(())
    }

    #[test]
    fn pinned_unicode_15_all_scalar_printability_repr_and_ascii_are_exact()
    -> Result<(), Box<dyn Error>> {
        let fixture = unicode_source_fixture()?;
        assert_eq!(fixture["environment"]["python"], "3.12.14");
        assert_eq!(fixture["environment"]["unicode_data"], "15.0.0");
        assert_eq!(
            fixture["source"]["sha256"],
            "bb01963178f28efc6e3a9578aecb89f53f8265e89c0518da92ec21c4955e85f1"
        );
        let mut bitmap = Sha256::new();
        let mut representation = Sha256::new();
        let mut ascii = Sha256::new();
        let mut count = 0u64;
        for codepoint in 0..=0x10ffff {
            let Some(character) = char::from_u32(codepoint) else {
                continue;
            };
            bitmap.update([u8::from(python_unicode_15_printable(character))]);
            for (hash, ascii_only) in [(&mut representation, false), (&mut ascii, true)] {
                let rendered = quoted(&character.to_string(), ascii_only);
                hash.update(codepoint.to_le_bytes());
                hash.update(u32::try_from(rendered.len())?.to_le_bytes());
                hash.update(rendered.as_bytes());
            }
            count += 1;
        }
        assert_eq!(count, 1_112_064);
        assert_eq!(fixture["scalar_scope"]["count"].as_u64(), Some(count));
        assert_eq!(
            format!("{:x}", bitmap.finalize()),
            fixture["scalar_scope"]["printability_bitmap_sha256"]
                .as_str()
                .ok_or("missing source printability SHA")?
        );
        assert_eq!(
            format!("{:x}", representation.finalize()),
            fixture["scalar_scope"]["repr_framed_sha256"]
                .as_str()
                .ok_or("missing source repr SHA")?
        );
        assert_eq!(
            format!("{:x}", ascii.finalize()),
            fixture["scalar_scope"]["ascii_framed_sha256"]
                .as_str()
                .ok_or("missing source ascii SHA")?
        );
        Ok(())
    }

    #[test]
    fn pinned_unicode_source_conversions_nested_quotes_and_width_are_exact()
    -> Result<(), Box<dyn Error>> {
        let fixture = unicode_source_fixture()?;
        let cases = fixture["cases"]
            .as_array()
            .ok_or("missing Unicode source cases")?;
        assert_eq!(cases.len(), 32);
        for case in cases {
            let value = case.get("value").ok_or("missing source value")?;
            let value = match value {
                Value::String(value) => primitive(NativePrimitive::String(value.clone())),
                value => NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value: value.clone(),
                },
            };
            assert_eq!(
                NativeTextFormatter::format(
                    case["template"].as_str().ok_or("missing source template")?,
                    &BTreeMap::from([("a".to_owned(), value)]),
                    &CancellationToken::default(),
                )?,
                case["output"].as_str().ok_or("missing original output")?,
                "original Unicode source case {}",
                case["id"],
            );
        }
        Ok(())
    }

    #[test]
    fn unicode_expansion_bounds_cancellation_and_recovery_are_atomic() -> Result<(), Box<dyn Error>>
    {
        let values = BTreeMap::from([(
            "a".to_owned(),
            primitive(NativePrimitive::String(
                "\u{e000}".repeat(NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES / 6 + 1),
            )),
        )]);
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert_eq!(
            NativeTextFormatter::format("{a!r}", &values, &cancellation),
            Err(NativeTextFormatError::Cancelled),
        );
        assert_eq!(
            NativeTextFormatter::format("{a!r}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::ResultTooLarge {
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES,
            }),
        );
        assert_eq!(
            NativeTextFormatter::format(
                "{a!r}|{a!a}",
                &BTreeMap::from([(
                    "a".to_owned(),
                    primitive(NativePrimitive::String("\u{2028}".to_owned()))
                )]),
                &CancellationToken::default(),
            )?,
            "'\\u2028'|'\\u2028'",
        );
        Ok(())
    }

    #[test]
    fn named_fields_escapes_conversions_and_nested_specs_are_exact() -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([
            (
                "a".to_owned(),
                primitive(NativePrimitive::String("café".to_owned())),
            ),
            ("b".to_owned(), primitive(NativePrimitive::Integer(42))),
            ("c".to_owned(), primitive(NativePrimitive::Integer(6))),
        ]);
        assert_eq!(
            NativeTextFormatter::format(
                "{{{a!r}}} {b:+0{c}x} {a!a}",
                &values,
                &CancellationToken::default(),
            )?,
            "{'café'} +0002a 'caf\\xe9'"
        );
        Ok(())
    }

    #[test]
    fn structured_item_and_attribute_access_are_checked() -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([(
            "a".to_owned(),
            NativeValue::PreservedUnknown {
                type_name: "record".to_owned(),
                value: serde_json::json!({"items": ["first", {"name": "second"}]}),
            },
        )]);
        assert_eq!(
            NativeTextFormatter::format(
                "{a[items][0]} {a[items][1][name]}",
                &values,
                &CancellationToken::default(),
            )?,
            "first second"
        );
        Ok(())
    }

    #[test]
    fn python_312_dictionary_attributes_do_not_alias_stored_keys() {
        let dictionary = serde_json::json!({
            "name": "stored name",
            "value": "stored value",
            "key": "stored key",
            "items": ["first", {"name": "second"}],
        });
        let values = BTreeMap::from([(
            "a".to_owned(),
            NativeValue::PreservedUnknown {
                type_name: "JSON".to_owned(),
                value: dictionary.clone(),
            },
        )]);
        for (template, field) in [
            ("{a.name}", "a.name"),
            ("{a.value}", "a.value"),
            ("{a.key}", "a.key"),
            ("{a.items[0]}", "a.items[0]"),
            ("{a[items][1].name}", "a[items][1].name"),
        ] {
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default()),
                Err(NativeTextFormatError::InvalidFieldAccess(field.to_owned())),
            );
        }
        for value in [
            NativeValue::PreservedUnknown {
                type_name: "JSON".to_owned(),
                value: serde_json::json!([dictionary.clone()]),
            },
            NativeValue::List {
                values: vec![NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value: dictionary,
                }],
            },
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    "{a[0].name}",
                    &BTreeMap::from([("a".to_owned(), value)]),
                    &CancellationToken::default(),
                ),
                Err(NativeTextFormatError::InvalidFieldAccess(
                    "a[0].name".to_owned()
                )),
            );
        }
    }

    #[test]
    fn python_312_dictionary_items_compose_with_conversion_specifiers_and_literal_keys()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([
            (
                "a".to_owned(),
                NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value: serde_json::json!({
                        "items": ["first", {"name": "second"}],
                        "name": "café",
                        "value": 1.0,
                        "name.with.dot": "literal dot",
                        "a b": "space key",
                    }),
                },
            ),
            ("b".to_owned(), primitive(NativePrimitive::Integer(3))),
        ]);
        for (template, expected) in [
            ("{a[items][0]} {a[items][1][name]}", "first second"),
            ("{a[name]!r:>10}", "    'café'"),
            ("{a[name]!a}", "'caf\\xe9'"),
            ("{a[items][1][name]:.{b}}", "sec"),
            ("{a[value]:.2}", "1.0"),
            ("{a[name.with.dot]}", "literal dot"),
            ("{a[a b]}", "space key"),
            ("{a[items]!s}", "['first', {'name': 'second'}]"),
            ("{a[items][1]!r}", "{'name': 'second'}"),
        ] {
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default())?,
                expected,
                "Python 3.12 dictionary item case {template}",
            );
        }
        Ok(())
    }

    #[test]
    fn dictionary_field_failures_keep_source_order_and_cancel_without_partial_output()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([(
            "a".to_owned(),
            NativeValue::PreservedUnknown {
                type_name: "JSON".to_owned(),
                value: serde_json::json!({"name": "café"}),
            },
        )]);
        for (template, error) in [
            (
                "{a.name} {b}",
                NativeTextFormatError::InvalidFieldAccess("a.name".to_owned()),
            ),
            (
                "{b} {a.name}",
                NativeTextFormatError::MissingField("b".to_owned()),
            ),
            (
                "{a[name]} {a.name}",
                NativeTextFormatError::InvalidFieldAccess("a.name".to_owned()),
            ),
            (
                "{a[missing]}",
                NativeTextFormatError::InvalidFieldAccess("a[missing]".to_owned()),
            ),
        ] {
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default()),
                Err(error),
            );
        }
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert_eq!(
            NativeTextFormatter::format("{a[name]} {a.name}", &values, &cancellation),
            Err(NativeTextFormatError::Cancelled),
        );
        assert_eq!(
            NativeTextFormatter::format("{a[name]}", &values, &CancellationToken::default())?,
            "café",
        );
        Ok(())
    }

    #[test]
    fn python_312_portable_numeric_attributes_preserve_source_scalar_types()
    -> Result<(), Box<dyn Error>> {
        for (primitive_value, outputs) in [
            (
                NativePrimitive::Boolean(true),
                [Some("1"), Some("0"), Some("1"), Some("1")],
            ),
            (
                NativePrimitive::Boolean(false),
                [Some("0"), Some("0"), Some("0"), Some("1")],
            ),
            (
                NativePrimitive::Integer(-7),
                [Some("-7"), Some("0"), Some("-7"), Some("1")],
            ),
            (
                NativePrimitive::Integer(0),
                [Some("0"), Some("0"), Some("0"), Some("1")],
            ),
            (
                NativePrimitive::Integer(i64::MIN),
                [
                    Some("-9223372036854775808"),
                    Some("0"),
                    Some("-9223372036854775808"),
                    Some("1"),
                ],
            ),
            (
                NativePrimitive::UnsignedInteger(u64::MAX),
                [
                    Some("18446744073709551615"),
                    Some("0"),
                    Some("18446744073709551615"),
                    Some("1"),
                ],
            ),
            (
                NativePrimitive::Number(1.25),
                [Some("1.25"), Some("0.0"), None, None],
            ),
            (
                NativePrimitive::Number(-0.0),
                [Some("-0.0"), Some("0.0"), None, None],
            ),
        ] {
            let values = BTreeMap::from([("a".to_owned(), primitive(primitive_value))]);
            for (name, expected) in ["real", "imag", "numerator", "denominator"]
                .into_iter()
                .zip(outputs)
            {
                let field = format!("a.{name}");
                let result = NativeTextFormatter::format(
                    &format!("{{{field}}}"),
                    &values,
                    &CancellationToken::default(),
                );
                match expected {
                    Some(expected) => assert_eq!(result?, expected),
                    None => assert_eq!(
                        result,
                        Err(NativeTextFormatError::InvalidFieldAccess(field))
                    ),
                }
            }
        }
        Ok(())
    }

    #[test]
    fn python_312_numeric_attributes_chain_and_format_through_existing_atoms()
    -> Result<(), Box<dyn Error>> {
        for (value, template, expected) in [
            (
                primitive(NativePrimitive::Boolean(true)),
                "{a.real!r}|{a.real:04d}",
                "1|0001",
            ),
            (
                primitive(NativePrimitive::Integer(-7)),
                "{a.real.numerator.denominator}|{a.imag.real}",
                "1|0",
            ),
            (
                primitive(NativePrimitive::Number(-0.0)),
                "{a.real.imag.real!r}|{a.imag.real!r}",
                "0.0|0.0",
            ),
            (
                primitive(NativePrimitive::Number(1.25)),
                "{a.real:.2}|{a.imag:.1}",
                "1.2|0e+00",
            ),
            (
                primitive(NativePrimitive::Integer(-7)),
                "{a.numerator!s}|{a.imag!r}|{a.denominator!a}",
                "-7|0|1",
            ),
            (
                NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value: serde_json::json!({"value": [true]}),
                },
                "{a[value][0].real}|{a[value][0].imag}|{a[value][0].numerator}|{a[value][0].denominator}",
                "1|0|1|1",
            ),
            (
                NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value: serde_json::json!({"value": [-0.0]}),
                },
                "{a[value][0].real!r}|{a[value][0].imag!r}",
                "-0.0|0.0",
            ),
            (
                NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value: serde_json::json!({"value": [{"n": u64::MAX}]}),
                },
                "{a[value][0][n].numerator:X}|{a[value][0][n].denominator:d}",
                "FFFFFFFFFFFFFFFF|1",
            ),
            (
                NativeValue::List {
                    values: vec![primitive(NativePrimitive::Number(-0.0))],
                },
                "{a[0].real!r}|{a[0].imag!r}",
                "-0.0|0.0",
            ),
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    template,
                    &BTreeMap::from([("a".to_owned(), value)]),
                    &CancellationToken::default(),
                )?,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn scalar_attributes_do_not_admit_dictionary_aliases_or_change_error_order() {
        for value in [
            NativeValue::PreservedUnknown {
                type_name: "JSON".to_owned(),
                value: serde_json::json!({"real": 1.0}),
            },
            primitive(NativePrimitive::String("7".to_owned())),
            primitive(NativePrimitive::Null),
            NativeValue::List {
                values: vec![primitive(NativePrimitive::Integer(7))],
            },
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    "{a.real}",
                    &BTreeMap::from([("a".to_owned(), value)]),
                    &CancellationToken::default(),
                ),
                Err(NativeTextFormatError::InvalidFieldAccess(
                    "a.real".to_owned()
                ))
            );
        }
        let values = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Number(1.25)))]);
        for (template, error) in [
            (
                "{a.real.imag.numerator}",
                NativeTextFormatError::InvalidFieldAccess("a.real.imag.numerator".to_owned()),
            ),
            (
                "{b}|{a.numerator}",
                NativeTextFormatError::MissingField("b".to_owned()),
            ),
            (
                "{a.numerator}|{b}",
                NativeTextFormatError::InvalidFieldAccess("a.numerator".to_owned()),
            ),
        ] {
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default()),
                Err(error)
            );
        }
    }

    #[test]
    fn scalar_attribute_formatting_keeps_capacity_cancellation_and_recovery_bounds()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Number(-0.0)))]);
        let precision = NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES + 1;
        let template = format!("{{a.real:.{precision}f}}");
        assert_eq!(
            NativeTextFormatter::format(&template, &values, &CancellationToken::default()),
            Err(NativeTextFormatError::ResultTooLarge {
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES
            })
        );
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert_eq!(
            NativeTextFormatter::format(&template, &values, &cancellation),
            Err(NativeTextFormatError::Cancelled)
        );
        assert_eq!(
            NativeTextFormatter::format(
                "{a.real!r}|{a.imag!r}",
                &values,
                &CancellationToken::default()
            )?,
            "-0.0|0.0"
        );
        Ok(())
    }

    #[test]
    fn python_312_ascii_decimal_dictionary_indices_never_alias_string_keys()
    -> Result<(), Box<dyn Error>> {
        let dictionary = serde_json::json!({
            "0": "stored:0", "1": "stored:1", "01": "stored:01", "0001": "stored:0001",
            "+1": "stored:+1", "-1": "stored:-1", " 1": "stored: 1", "1 ": "stored:1 ",
            "1_0": "stored:1_0", "": "stored:",
            "999999999999999999999999999999": "stored:overflow",
            "name.with.dot": "literal",
        });
        let values = BTreeMap::from([(
            "a".to_owned(),
            NativeValue::PreservedUnknown {
                type_name: "JSON".to_owned(),
                value: dictionary,
            },
        )]);
        for key in ["0", "1", "01", "0001", "", "999999999999999999999999999999"] {
            let field = format!("a[{key}]");
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{{field}}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidFieldAccess(field))
            );
        }
        for key in ["+1", "-1", " 1", "1 ", "1_0"] {
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a[{key}]}}"),
                    &values,
                    &CancellationToken::default()
                )?,
                format!("stored:{key}")
            );
        }
        assert_eq!(
            NativeTextFormatter::format(
                "{a[+1]!r}|{a[-1]!s}",
                &values,
                &CancellationToken::default()
            )?,
            "'stored:+1'|stored:-1"
        );
        assert_eq!(
            NativeTextFormatter::format(
                "{a[name.with.dot]}",
                &values,
                &CancellationToken::default()
            )?,
            "literal"
        );
        Ok(())
    }

    #[test]
    fn python_312_sequence_indices_require_decimal_bytes_and_preserve_unicode_characters()
    -> Result<(), Box<dyn Error>> {
        for (value, expected) in [
            (
                NativeValue::List {
                    values: vec![
                        primitive(NativePrimitive::String("zero".to_owned())),
                        primitive(NativePrimitive::String("one".to_owned())),
                    ],
                },
                ["zero", "one", "one", "one"],
            ),
            (
                NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value: serde_json::json!(["zero", "one"]),
                },
                ["zero", "one", "one", "one"],
            ),
            (
                primitive(NativePrimitive::String("é😀".to_owned())),
                ["é", "😀", "😀", "😀"],
            ),
            (
                NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value: serde_json::json!("é😀"),
                },
                ["é", "😀", "😀", "😀"],
            ),
        ] {
            let values = BTreeMap::from([("a".to_owned(), value)]);
            for (key, expected) in ["0", "1", "01", "0001"].into_iter().zip(expected) {
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{a[{key}]}}"),
                        &values,
                        &CancellationToken::default()
                    )?,
                    expected
                );
            }
            for key in [
                "2",
                "+1",
                "-1",
                " 1",
                "1 ",
                "1_0",
                "",
                "999999999999999999999999999999",
            ] {
                let field = format!("a[{key}]");
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{{field}}}"),
                        &values,
                        &CancellationToken::default()
                    ),
                    Err(NativeTextFormatError::InvalidFieldAccess(field))
                );
            }
        }
        let values = BTreeMap::from([(
            "a".to_owned(),
            NativeValue::PreservedUnknown {
                type_name: "JSON".to_owned(),
                value: serde_json::json!([{"name": "zero"}, {"name": "one"}]),
            },
        )]);
        assert_eq!(
            NativeTextFormatter::format("{a[01][name]}", &values, &CancellationToken::default())?,
            "one"
        );
        let values = BTreeMap::from([(
            "a".to_owned(),
            NativeValue::PreservedUnknown {
                type_name: "JSON".to_owned(),
                value: serde_json::json!({"text": "é😀"}),
            },
        )]);
        assert_eq!(
            NativeTextFormatter::format("{a[text][01]!r}", &values, &CancellationToken::default())?,
            "'😀'"
        );
        Ok(())
    }

    #[test]
    fn typed_dictionary_index_errors_keep_first_failure_and_no_partial_result() {
        let values = BTreeMap::from([(
            "a".to_owned(),
            NativeValue::PreservedUnknown {
                type_name: "JSON".to_owned(),
                value: serde_json::json!({"0": "string zero"}),
            },
        )]);
        for (template, error) in [
            (
                "{a[0]} {b}",
                NativeTextFormatError::InvalidFieldAccess("a[0]".to_owned()),
            ),
            (
                "{b} {a[0]}",
                NativeTextFormatError::MissingField("b".to_owned()),
            ),
        ] {
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default()),
                Err(error)
            );
        }
    }

    #[test]
    fn typed_index_lookup_reuses_capacity_cancellation_and_recovery_bounds()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([(
            "a".to_owned(),
            primitive(NativePrimitive::String("é😀".to_owned())),
        )]);
        let template = format!("{{a[01]:>{}}}", NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES + 1);
        assert_eq!(
            NativeTextFormatter::format(&template, &values, &CancellationToken::default()),
            Err(NativeTextFormatError::ResultTooLarge {
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES
            })
        );
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert_eq!(
            NativeTextFormatter::format("{a[+1]}", &values, &cancellation),
            Err(NativeTextFormatError::Cancelled)
        );
        assert_eq!(
            NativeTextFormatter::format("{a[01]!r}", &values, &CancellationToken::default())?,
            "'😀'"
        );
        Ok(())
    }

    #[test]
    fn pinned_python_unicode_15_decimal_values_match_every_scalar() {
        let mut authority = Sha256::new();
        let mut scalar_count = 0usize;
        let mut decimal_count = 0usize;
        for codepoint in 0..=0x10ffff_u32 {
            let Some(character) = char::from_u32(codepoint) else {
                continue;
            };
            let decimal = python_unicode_15_decimal_digit(character);
            authority.update(codepoint.to_le_bytes());
            authority.update([decimal.unwrap_or(u8::MAX)]);
            scalar_count += 1;
            decimal_count += usize::from(decimal.is_some());
        }
        assert_eq!(scalar_count, 1_112_064);
        assert_eq!(decimal_count, 680);
        assert_eq!(PYTHON_UNICODE_15_DECIMAL_ZEROES.len(), 68);
        assert_eq!(
            format!("{:x}", authority.finalize()),
            "b07ba3cbd8dd2ebbcff70fa65ae1a536c9231a77a8a0913fcb344ab978f6d708"
        );
    }

    #[test]
    fn python_312_all_decimal_codepoints_select_lists_strings_and_reject_string_dictionary_keys()
    -> Result<(), Box<dyn Error>> {
        let list = NativeValue::List {
            values: (0..10)
                .map(|index| primitive(NativePrimitive::String(format!("v{index}"))))
                .collect(),
        };
        let json_list = NativeValue::PreservedUnknown {
            type_name: "JSON".to_owned(),
            value: Value::Array(
                (0..10)
                    .map(|index| Value::String(format!("v{index}")))
                    .collect(),
            ),
        };
        let string = primitive(NativePrimitive::String("0123456789".to_owned()));
        let json_string = NativeValue::PreservedUnknown {
            type_name: "JSON".to_owned(),
            value: Value::String("0123456789".to_owned()),
        };
        let mixed_list = NativeValue::List {
            values: (0..100)
                .map(|index| primitive(NativePrimitive::String(format!("v{index}"))))
                .collect(),
        };
        let nested_string = NativeValue::PreservedUnknown {
            type_name: "JSON".to_owned(),
            value: serde_json::json!({"text": "0123456789"}),
        };
        for zero in PYTHON_UNICODE_15_DECIMAL_ZEROES {
            for digit in 0..10 {
                let character =
                    char::from_u32(zero + digit).ok_or("invalid source decimal scalar")?;
                let key = character.to_string();
                for (value, expected) in [
                    (list.clone(), format!("v{digit}")),
                    (json_list.clone(), format!("v{digit}")),
                    (string.clone(), digit.to_string()),
                    (json_string.clone(), digit.to_string()),
                ] {
                    assert_eq!(
                        NativeTextFormatter::format(
                            &format!("{{a[{key}]}}"),
                            &BTreeMap::from([("a".to_owned(), value)]),
                            &CancellationToken::default()
                        )?,
                        expected
                    );
                }
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{a[text][{key}]!r}}"),
                        &BTreeMap::from([("a".to_owned(), nested_string.clone())]),
                        &CancellationToken::default()
                    )?,
                    format!("'{digit}'")
                );
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{a[{key}0]}}"),
                        &BTreeMap::from([("a".to_owned(), mixed_list.clone())]),
                        &CancellationToken::default()
                    )?,
                    format!("v{}", digit * 10)
                );
                let dictionary = Value::Object(serde_json::Map::from_iter([(
                    key.clone(),
                    Value::String("stored".to_owned()),
                )]));
                let field = format!("a[{key}]");
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{{field}}}"),
                        &BTreeMap::from([(
                            "a".to_owned(),
                            NativeValue::PreservedUnknown {
                                type_name: "JSON".to_owned(),
                                value: dictionary.clone()
                            }
                        )]),
                        &CancellationToken::default()
                    ),
                    Err(NativeTextFormatError::InvalidFieldAccess(field))
                );
                let field = format!("a[map][{key}]");
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{{field}}}"),
                        &BTreeMap::from([(
                            "a".to_owned(),
                            NativeValue::PreservedUnknown {
                                type_name: "JSON".to_owned(),
                                value: serde_json::json!({"map": dictionary})
                            }
                        )]),
                        &CancellationToken::default()
                    ),
                    Err(NativeTextFormatError::InvalidFieldAccess(field))
                );
                let text_key = format!("+{key}");
                let dictionary = Value::Object(serde_json::Map::from_iter([(
                    text_key.clone(),
                    Value::String("stored".to_owned()),
                )]));
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{a[{text_key}]}}"),
                        &BTreeMap::from([(
                            "a".to_owned(),
                            NativeValue::PreservedUnknown {
                                type_name: "JSON".to_owned(),
                                value: dictionary
                            }
                        )]),
                        &CancellationToken::default()
                    )?,
                    "stored"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn python_312_nondecimal_numeric_controls_remain_exact_text_keys() -> Result<(), Box<dyn Error>>
    {
        let controls = [
            0xb2, 0xb3, 0xb9, 0xbc, 0xbd, 0xbe, 0x9f4, 0x9f5, 0x9f6, 0x9f7, 0x9f8, 0x9f9, 0xb72,
            0xb73, 0xb74, 0xb75, 0xb76, 0xb77, 0xbf0, 0xbf1, 0xbf2, 0xc78, 0xc79, 0xc7a, 0xc7b,
            0xc7c, 0xc7d, 0xc7e, 0xd58, 0xd59, 0xd5a, 0xd5b, 0xd5c, 0xd5d, 0xd5e, 0xd70, 0xd71,
            0xd72, 0xd73, 0xd74, 0xd75, 0xd76, 0xd77, 0xd78, 0xf2a, 0xf2b, 0xf2c, 0xf2d, 0xf2e,
            0xf2f, 0xf30, 0xf31, 0xf32, 0xf33, 0x1369, 0x136a, 0x136b, 0x136c, 0x136d, 0x136e,
            0x136f, 0x1370, 0x1371, 0x1372, 0x1373, 0x1374, 0x1375, 0x1376, 0x1377, 0x1378, 0x1379,
            0x137a, 0x137b, 0x137c, 0x16ee, 0x16ef, 0x16f0, 0x17f0, 0x17f1, 0x17f2, 0x17f3, 0x17f4,
            0x17f5, 0x17f6, 0x17f7, 0x17f8, 0x17f9, 0x19da, 0x2070, 0x2074, 0x2075, 0x2076, 0x2077,
            0x2078, 0x2079, 0x2080, 0x2081, 0x2082, 0x2083, 0x2084, 0x2085, 0x2086, 0x2087, 0x2088,
            0x2089, 0x2150, 0x2151, 0x2152, 0x2153, 0x2154, 0x2155, 0x2156, 0x2157, 0x2158, 0x2159,
            0x215a, 0x215b, 0x215c, 0x215d, 0x215e, 0x215f, 0x2160, 0x2161, 0x2162, 0x2163, 0x2164,
            0x2165, 0x2166, 0x2167, 0x2168, 0x2169, 0x216a, 0x216b, 0x216c, 0x216d, 0x216e, 0x216f,
            0x2170, 0x2171, 0x2172, 0x2173, 0x2174, 0x2175, 0x2176, 0x2177, 0x2178, 0x2179, 0x217a,
            0x217b, 0x217c, 0x217d, 0x217e, 0x217f, 0x2180, 0x2181, 0x2182, 0x2185, 0x2186, 0x2187,
            0x2188, 0x2189, 0x2460, 0x2461, 0x2462, 0x2463, 0x2464, 0x2465, 0x2466, 0x2467, 0x2468,
            0x2469, 0x246a, 0x246b, 0x246c, 0x246d, 0x246e, 0x246f, 0x2470, 0x2471, 0x2472, 0x2473,
            0x2474, 0x2475, 0x2476, 0x2477, 0x2478, 0x2479, 0x247a, 0x247b, 0x247c, 0x247d, 0x247e,
            0x247f, 0x2480, 0x2481, 0x2482, 0x2483, 0x2484, 0x2485, 0x2486, 0x2487, 0x2488, 0x2489,
            0x248a, 0x248b, 0x248c, 0x248d, 0x248e, 0x248f, 0x2490, 0x2491, 0x2492, 0x2493, 0x2494,
            0x2495, 0x2496, 0x2497, 0x2498, 0x2499, 0x249a, 0x249b, 0x24ea, 0x24eb, 0x24ec, 0x24ed,
            0x24ee, 0x24ef, 0x24f0, 0x24f1, 0x24f2, 0x24f3, 0x24f4, 0x24f5, 0x24f6, 0x24f7, 0x24f8,
            0x24f9, 0x24fa, 0x24fb, 0x24fc, 0x24fd, 0x24fe, 0x24ff, 0x2776, 0x2777, 0x2778, 0x2779,
            0x277a, 0x277b, 0x277c, 0x277d, 0x277e, 0x277f, 0x2780, 0x2781, 0x2782, 0x2783, 0x2784,
            0x2785, 0x2786, 0x2787, 0x2788, 0x2789, 0x278a, 0x278b, 0x278c, 0x278d, 0x278e, 0x278f,
            0x2790, 0x2791, 0x2792, 0x2793, 0x2cfd, 0x3007, 0x3021, 0x3022, 0x3023, 0x3024, 0x3025,
            0x3026, 0x3027, 0x3028, 0x3029, 0x3038, 0x3039, 0x303a, 0x3192, 0x3193, 0x3194, 0x3195,
            0x3220, 0x3221, 0x3222, 0x3223, 0x3224, 0x3225, 0x3226, 0x3227, 0x3228, 0x3229, 0x3248,
            0x3249, 0x324a, 0x324b, 0x324c, 0x324d, 0x324e, 0x324f, 0x3251, 0x3252, 0x3253, 0x3254,
            0x3255, 0x3256, 0x3257, 0x3258, 0x3259, 0x325a, 0x325b, 0x325c, 0x325d, 0x325e, 0x325f,
            0x3280, 0x3281, 0x3282, 0x3283, 0x3284, 0x3285, 0x3286, 0x3287, 0x3288, 0x3289, 0x32b1,
            0x32b2, 0x32b3, 0x32b4, 0x32b5, 0x32b6, 0x32b7, 0x32b8, 0x32b9, 0x32ba, 0x32bb, 0x32bc,
            0x32bd, 0x32be, 0x32bf, 0x3405, 0x3483, 0x382a, 0x3b4d, 0x4e00, 0x4e03, 0x4e07, 0x4e09,
            0x4e5d, 0x4e8c, 0x4e94, 0x4e96, 0x4ebf, 0x4ec0, 0x4edf, 0x4ee8, 0x4f0d, 0x4f70, 0x5104,
            0x5146, 0x5169, 0x516b, 0x516d, 0x5341, 0x5343, 0x5344, 0x5345, 0x534c, 0x53c1, 0x53c2,
            0x53c3, 0x53c4, 0x56db, 0x58f1, 0x58f9, 0x5e7a, 0x5efe, 0x5eff, 0x5f0c, 0x5f0d, 0x5f0e,
            0x5f10, 0x62fe, 0x634c, 0x67d2, 0x6f06, 0x7396, 0x767e, 0x8086, 0x842c, 0x8cae, 0x8cb3,
            0x8d30, 0x9621, 0x9646, 0x964c, 0x9678, 0x96f6, 0xa6e6, 0xa6e7, 0xa6e8, 0xa6e9, 0xa6ea,
            0xa6eb, 0xa6ec, 0xa6ed, 0xa6ee, 0xa6ef, 0xa830, 0xa831, 0xa832, 0xa833, 0xa834, 0xa835,
            0xf96b, 0xf973, 0xf978, 0xf9b2, 0xf9d1, 0xf9d3, 0xf9fd, 0x10107, 0x10108, 0x10109,
            0x1010a, 0x1010b, 0x1010c, 0x1010d, 0x1010e, 0x1010f, 0x10110, 0x10111, 0x10112,
            0x10113, 0x10114, 0x10115, 0x10116, 0x10117, 0x10118, 0x10119, 0x1011a, 0x1011b,
            0x1011c, 0x1011d, 0x1011e, 0x1011f, 0x10120, 0x10121, 0x10122, 0x10123, 0x10124,
            0x10125, 0x10126, 0x10127, 0x10128, 0x10129, 0x1012a, 0x1012b, 0x1012c, 0x1012d,
            0x1012e, 0x1012f, 0x10130, 0x10131, 0x10132, 0x10133, 0x10140, 0x10141, 0x10142,
            0x10143, 0x10144, 0x10145, 0x10146, 0x10147, 0x10148, 0x10149, 0x1014a, 0x1014b,
            0x1014c, 0x1014d, 0x1014e, 0x1014f, 0x10150, 0x10151, 0x10152, 0x10153, 0x10154,
            0x10155, 0x10156, 0x10157, 0x10158, 0x10159, 0x1015a, 0x1015b, 0x1015c, 0x1015d,
            0x1015e, 0x1015f, 0x10160, 0x10161, 0x10162, 0x10163, 0x10164, 0x10165, 0x10166,
            0x10167, 0x10168, 0x10169, 0x1016a, 0x1016b, 0x1016c, 0x1016d, 0x1016e, 0x1016f,
            0x10170, 0x10171, 0x10172, 0x10173, 0x10174, 0x10175, 0x10176, 0x10177, 0x10178,
            0x1018a, 0x1018b, 0x102e1, 0x102e2, 0x102e3, 0x102e4, 0x102e5, 0x102e6, 0x102e7,
            0x102e8, 0x102e9, 0x102ea, 0x102eb, 0x102ec, 0x102ed, 0x102ee, 0x102ef, 0x102f0,
            0x102f1, 0x102f2, 0x102f3, 0x102f4, 0x102f5, 0x102f6, 0x102f7, 0x102f8, 0x102f9,
            0x102fa, 0x102fb, 0x10320, 0x10321, 0x10322, 0x10323, 0x10341, 0x1034a, 0x103d1,
            0x103d2, 0x103d3, 0x103d4, 0x103d5, 0x10858, 0x10859, 0x1085a, 0x1085b, 0x1085c,
            0x1085d, 0x1085e, 0x1085f, 0x10879, 0x1087a, 0x1087b, 0x1087c, 0x1087d, 0x1087e,
            0x1087f, 0x108a7, 0x108a8, 0x108a9, 0x108aa, 0x108ab, 0x108ac, 0x108ad, 0x108ae,
            0x108af, 0x108fb, 0x108fc, 0x108fd, 0x108fe, 0x108ff, 0x10916, 0x10917, 0x10918,
            0x10919, 0x1091a, 0x1091b, 0x109bc, 0x109bd, 0x109c0, 0x109c1, 0x109c2, 0x109c3,
            0x109c4, 0x109c5, 0x109c6, 0x109c7, 0x109c8, 0x109c9, 0x109ca, 0x109cb, 0x109cc,
            0x109cd, 0x109ce, 0x109cf, 0x109d2, 0x109d3, 0x109d4, 0x109d5, 0x109d6, 0x109d7,
            0x109d8, 0x109d9, 0x109da, 0x109db, 0x109dc, 0x109dd, 0x109de, 0x109df, 0x109e0,
            0x109e1, 0x109e2, 0x109e3, 0x109e4, 0x109e5, 0x109e6, 0x109e7, 0x109e8, 0x109e9,
            0x109ea, 0x109eb, 0x109ec, 0x109ed, 0x109ee, 0x109ef, 0x109f0, 0x109f1, 0x109f2,
            0x109f3, 0x109f4, 0x109f5, 0x109f6, 0x109f7, 0x109f8, 0x109f9, 0x109fa, 0x109fb,
            0x109fc, 0x109fd, 0x109fe, 0x109ff, 0x10a40, 0x10a41, 0x10a42, 0x10a43, 0x10a44,
            0x10a45, 0x10a46, 0x10a47, 0x10a48, 0x10a7d, 0x10a7e, 0x10a9d, 0x10a9e, 0x10a9f,
            0x10aeb, 0x10aec, 0x10aed, 0x10aee, 0x10aef, 0x10b58, 0x10b59, 0x10b5a, 0x10b5b,
            0x10b5c, 0x10b5d, 0x10b5e, 0x10b5f, 0x10b78, 0x10b79, 0x10b7a, 0x10b7b, 0x10b7c,
            0x10b7d, 0x10b7e, 0x10b7f, 0x10ba9, 0x10baa, 0x10bab, 0x10bac, 0x10bad, 0x10bae,
            0x10baf, 0x10cfa, 0x10cfb, 0x10cfc, 0x10cfd, 0x10cfe, 0x10cff, 0x10e60, 0x10e61,
            0x10e62, 0x10e63, 0x10e64, 0x10e65, 0x10e66, 0x10e67, 0x10e68, 0x10e69, 0x10e6a,
            0x10e6b, 0x10e6c, 0x10e6d, 0x10e6e, 0x10e6f, 0x10e70, 0x10e71, 0x10e72, 0x10e73,
            0x10e74, 0x10e75, 0x10e76, 0x10e77, 0x10e78, 0x10e79, 0x10e7a, 0x10e7b, 0x10e7c,
            0x10e7d, 0x10e7e, 0x10f1d, 0x10f1e, 0x10f1f, 0x10f20, 0x10f21, 0x10f22, 0x10f23,
            0x10f24, 0x10f25, 0x10f26, 0x10f51, 0x10f52, 0x10f53, 0x10f54, 0x10fc5, 0x10fc6,
            0x10fc7, 0x10fc8, 0x10fc9, 0x10fca, 0x10fcb, 0x11052, 0x11053, 0x11054, 0x11055,
            0x11056, 0x11057, 0x11058, 0x11059, 0x1105a, 0x1105b, 0x1105c, 0x1105d, 0x1105e,
            0x1105f, 0x11060, 0x11061, 0x11062, 0x11063, 0x11064, 0x11065, 0x111e1, 0x111e2,
            0x111e3, 0x111e4, 0x111e5, 0x111e6, 0x111e7, 0x111e8, 0x111e9, 0x111ea, 0x111eb,
            0x111ec, 0x111ed, 0x111ee, 0x111ef, 0x111f0, 0x111f1, 0x111f2, 0x111f3, 0x111f4,
            0x1173a, 0x1173b, 0x118ea, 0x118eb, 0x118ec, 0x118ed, 0x118ee, 0x118ef, 0x118f0,
            0x118f1, 0x118f2, 0x11c5a, 0x11c5b, 0x11c5c, 0x11c5d, 0x11c5e, 0x11c5f, 0x11c60,
            0x11c61, 0x11c62, 0x11c63, 0x11c64, 0x11c65, 0x11c66, 0x11c67, 0x11c68, 0x11c69,
            0x11c6a, 0x11c6b, 0x11c6c, 0x11fc0, 0x11fc1, 0x11fc2, 0x11fc3, 0x11fc4, 0x11fc5,
            0x11fc6, 0x11fc7, 0x11fc8, 0x11fc9, 0x11fca, 0x11fcb, 0x11fcc, 0x11fcd, 0x11fce,
            0x11fcf, 0x11fd0, 0x11fd1, 0x11fd2, 0x11fd3, 0x11fd4, 0x12400, 0x12401, 0x12402,
            0x12403, 0x12404, 0x12405, 0x12406, 0x12407, 0x12408, 0x12409, 0x1240a, 0x1240b,
            0x1240c, 0x1240d, 0x1240e, 0x1240f, 0x12410, 0x12411, 0x12412, 0x12413, 0x12414,
            0x12415, 0x12416, 0x12417, 0x12418, 0x12419, 0x1241a, 0x1241b, 0x1241c, 0x1241d,
            0x1241e, 0x1241f, 0x12420, 0x12421, 0x12422, 0x12423, 0x12424, 0x12425, 0x12426,
            0x12427, 0x12428, 0x12429, 0x1242a, 0x1242b, 0x1242c, 0x1242d, 0x1242e, 0x1242f,
            0x12430, 0x12431, 0x12432, 0x12433, 0x12434, 0x12435, 0x12436, 0x12437, 0x12438,
            0x12439, 0x1243a, 0x1243b, 0x1243c, 0x1243d, 0x1243e, 0x1243f, 0x12440, 0x12441,
            0x12442, 0x12443, 0x12444, 0x12445, 0x12446, 0x12447, 0x12448, 0x12449, 0x1244a,
            0x1244b, 0x1244c, 0x1244d, 0x1244e, 0x1244f, 0x12450, 0x12451, 0x12452, 0x12453,
            0x12454, 0x12455, 0x12456, 0x12457, 0x12458, 0x12459, 0x1245a, 0x1245b, 0x1245c,
            0x1245d, 0x1245e, 0x1245f, 0x12460, 0x12461, 0x12462, 0x12463, 0x12464, 0x12465,
            0x12466, 0x12467, 0x12468, 0x12469, 0x1246a, 0x1246b, 0x1246c, 0x1246d, 0x1246e,
            0x16b5b, 0x16b5c, 0x16b5d, 0x16b5e, 0x16b5f, 0x16b60, 0x16b61, 0x16e80, 0x16e81,
            0x16e82, 0x16e83, 0x16e84, 0x16e85, 0x16e86, 0x16e87, 0x16e88, 0x16e89, 0x16e8a,
            0x16e8b, 0x16e8c, 0x16e8d, 0x16e8e, 0x16e8f, 0x16e90, 0x16e91, 0x16e92, 0x16e93,
            0x16e94, 0x16e95, 0x16e96, 0x1d2c0, 0x1d2c1, 0x1d2c2, 0x1d2c3, 0x1d2c4, 0x1d2c5,
            0x1d2c6, 0x1d2c7, 0x1d2c8, 0x1d2c9, 0x1d2ca, 0x1d2cb, 0x1d2cc, 0x1d2cd, 0x1d2ce,
            0x1d2cf, 0x1d2d0, 0x1d2d1, 0x1d2d2, 0x1d2d3, 0x1d2e0, 0x1d2e1, 0x1d2e2, 0x1d2e3,
            0x1d2e4, 0x1d2e5, 0x1d2e6, 0x1d2e7, 0x1d2e8, 0x1d2e9, 0x1d2ea, 0x1d2eb, 0x1d2ec,
            0x1d2ed, 0x1d2ee, 0x1d2ef, 0x1d2f0, 0x1d2f1, 0x1d2f2, 0x1d2f3, 0x1d360, 0x1d361,
            0x1d362, 0x1d363, 0x1d364, 0x1d365, 0x1d366, 0x1d367, 0x1d368, 0x1d369, 0x1d36a,
            0x1d36b, 0x1d36c, 0x1d36d, 0x1d36e, 0x1d36f, 0x1d370, 0x1d371, 0x1d372, 0x1d373,
            0x1d374, 0x1d375, 0x1d376, 0x1d377, 0x1d378, 0x1e8c7, 0x1e8c8, 0x1e8c9, 0x1e8ca,
            0x1e8cb, 0x1e8cc, 0x1e8cd, 0x1e8ce, 0x1e8cf, 0x1ec71, 0x1ec72, 0x1ec73, 0x1ec74,
            0x1ec75, 0x1ec76, 0x1ec77, 0x1ec78, 0x1ec79, 0x1ec7a, 0x1ec7b, 0x1ec7c, 0x1ec7d,
            0x1ec7e, 0x1ec7f, 0x1ec80, 0x1ec81, 0x1ec82, 0x1ec83, 0x1ec84, 0x1ec85, 0x1ec86,
            0x1ec87, 0x1ec88, 0x1ec89, 0x1ec8a, 0x1ec8b, 0x1ec8c, 0x1ec8d, 0x1ec8e, 0x1ec8f,
            0x1ec90, 0x1ec91, 0x1ec92, 0x1ec93, 0x1ec94, 0x1ec95, 0x1ec96, 0x1ec97, 0x1ec98,
            0x1ec99, 0x1ec9a, 0x1ec9b, 0x1ec9c, 0x1ec9d, 0x1ec9e, 0x1ec9f, 0x1eca0, 0x1eca1,
            0x1eca2, 0x1eca3, 0x1eca4, 0x1eca5, 0x1eca6, 0x1eca7, 0x1eca8, 0x1eca9, 0x1ecaa,
            0x1ecab, 0x1ecad, 0x1ecae, 0x1ecaf, 0x1ecb1, 0x1ecb2, 0x1ecb3, 0x1ecb4, 0x1ed01,
            0x1ed02, 0x1ed03, 0x1ed04, 0x1ed05, 0x1ed06, 0x1ed07, 0x1ed08, 0x1ed09, 0x1ed0a,
            0x1ed0b, 0x1ed0c, 0x1ed0d, 0x1ed0e, 0x1ed0f, 0x1ed10, 0x1ed11, 0x1ed12, 0x1ed13,
            0x1ed14, 0x1ed15, 0x1ed16, 0x1ed17, 0x1ed18, 0x1ed19, 0x1ed1a, 0x1ed1b, 0x1ed1c,
            0x1ed1d, 0x1ed1e, 0x1ed1f, 0x1ed20, 0x1ed21, 0x1ed22, 0x1ed23, 0x1ed24, 0x1ed25,
            0x1ed26, 0x1ed27, 0x1ed28, 0x1ed29, 0x1ed2a, 0x1ed2b, 0x1ed2c, 0x1ed2d, 0x1ed2f,
            0x1ed30, 0x1ed31, 0x1ed32, 0x1ed33, 0x1ed34, 0x1ed35, 0x1ed36, 0x1ed37, 0x1ed38,
            0x1ed39, 0x1ed3a, 0x1ed3b, 0x1ed3c, 0x1ed3d, 0x1f100, 0x1f101, 0x1f102, 0x1f103,
            0x1f104, 0x1f105, 0x1f106, 0x1f107, 0x1f108, 0x1f109, 0x1f10a, 0x1f10b, 0x1f10c,
            0x20001, 0x20064, 0x200e2, 0x20121, 0x2092a, 0x20983, 0x2098c, 0x2099c, 0x20aea,
            0x20afd, 0x20b19, 0x22390, 0x22998, 0x23b1b, 0x2626d, 0x2f890,
        ];
        assert_eq!(controls.len(), 1232);
        for codepoint in controls.into_iter().chain([0x10d40, 0x1e5f1]) {
            let character = char::from_u32(codepoint).ok_or("invalid source numeric control")?;
            assert_eq!(python_unicode_15_decimal_digit(character), None);
            let key = character.to_string();
            let field = format!("a[{key}]");
            let list = NativeValue::List {
                values: vec![primitive(NativePrimitive::String("v0".to_owned()))],
            };
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{{field}}}"),
                    &BTreeMap::from([("a".to_owned(), list)]),
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidFieldAccess(field))
            );
            let dictionary = Value::Object(serde_json::Map::from_iter([(
                key.clone(),
                Value::String("stored".to_owned()),
            )]));
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a[{key}]}}"),
                    &BTreeMap::from([(
                        "a".to_owned(),
                        NativeValue::PreservedUnknown {
                            type_name: "JSON".to_owned(),
                            value: dictionary
                        }
                    )]),
                    &CancellationToken::default()
                )?,
                "stored"
            );
        }
        Ok(())
    }

    #[test]
    fn python_312_unicode_decimal_overflow_precedes_later_text_and_keeps_bounds()
    -> Result<(), Box<dyn Error>> {
        assert_eq!(
            python_decimal_field_index(&isize::MAX.to_string(), "a[index]")?,
            Some(isize::MAX as usize)
        );
        let source_overflow = ((isize::MAX as u128) + 1).to_string();
        for key in [
            source_overflow,
            "9".repeat(40),
            format!("{}x", "9".repeat(40)),
            format!("{}²", "9".repeat(40)),
            "٩".repeat(40),
            format!("{}x", "٩".repeat(40)),
            format!("١{}", "٠".repeat(19)),
        ] {
            assert_eq!(
                python_decimal_field_index(&key, "a[index]"),
                Err(NativeTextFormatError::InvalidFieldAccess(
                    "a[index]".to_owned()
                ))
            );
            let field = format!("a[{key}]");
            let dictionary = Value::Object(serde_json::Map::from_iter([(
                key.clone(),
                Value::String("stored".to_owned()),
            )]));
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{{field}}}"),
                    &BTreeMap::from([(
                        "a".to_owned(),
                        NativeValue::PreservedUnknown {
                            type_name: "JSON".to_owned(),
                            value: dictionary
                        }
                    )]),
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidFieldAccess(field))
            );
        }
        for key in [
            format!("{}x", "٠".repeat(40)),
            format!("x{}", "9".repeat(40)),
            format!("+{}", "٩".repeat(40)),
        ] {
            assert_eq!(python_decimal_field_index(&key, "a[index]")?, None);
        }
        let key = format!("{}1", "٠".repeat(80));
        assert_eq!(python_decimal_field_index(&key, "a[index]")?, Some(1));
        let values = BTreeMap::from([(
            "a".to_owned(),
            primitive(NativePrimitive::String("0123456789".to_owned())),
        )]);
        assert_eq!(
            NativeTextFormatter::format(
                &format!("{{a[{key}]}}"),
                &values,
                &CancellationToken::default()
            )?,
            "1"
        );
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert_eq!(
            NativeTextFormatter::format("{a[٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩٩]}", &values, &cancellation),
            Err(NativeTextFormatError::Cancelled)
        );
        let template = format!("{{a[١]:>{}}}", NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES + 1);
        assert_eq!(
            NativeTextFormatter::format(&template, &values, &CancellationToken::default()),
            Err(NativeTextFormatError::ResultTooLarge {
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES
            })
        );
        assert_eq!(
            NativeTextFormatter::format("{a[١]!r}", &values, &CancellationToken::default())?,
            "'1'"
        );
        Ok(())
    }

    #[test]
    fn python_numeric_format_language_is_source_compatible() -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([
            ("a".to_owned(), primitive(NativePrimitive::Integer(42))),
            ("b".to_owned(), primitive(NativePrimitive::Integer(12_345))),
            ("c".to_owned(), primitive(NativePrimitive::Integer(65))),
            ("d".to_owned(), primitive(NativePrimitive::Number(1.0))),
            ("e".to_owned(), primitive(NativePrimitive::Number(12_345.0))),
            ("f".to_owned(), primitive(NativePrimitive::Number(12.34))),
            ("g".to_owned(), primitive(NativePrimitive::Boolean(true))),
        ]);
        assert_eq!(
            NativeTextFormatter::format(
                "{a:=+8d}|{b:,d}|{b:_d}|{c:c}|{d:.2e}|{e:.3g}|{f:.2}|{d:#.0f}|{d:#.3g}|{g:d}|{g:>8}",
                &values,
                &CancellationToken::default(),
            )?,
            "+     42|12,345|12_345|A|1.00e+00|1.23e+04|1.2e+01|1.|1.00|1|       1"
        );
        Ok(())
    }

    #[test]
    fn python_alignment_grouping_alternate_and_type_matrix_is_exact() -> Result<(), Box<dyn Error>>
    {
        let values = BTreeMap::from([
            ("i".to_owned(), primitive(NativePrimitive::Integer(42))),
            ("n".to_owned(), primitive(NativePrimitive::Integer(12_345))),
            ("x".to_owned(), primitive(NativePrimitive::Integer(255))),
            ("z".to_owned(), primitive(NativePrimitive::Number(-0.0))),
            ("v".to_owned(), primitive(NativePrimitive::Number(1_234.5))),
        ]);
        assert_eq!(
            NativeTextFormatter::format(
                "{i:<8d}|{i:^8d}|{i:*>8d}|{i:+d}|{i: d}|{x:#b}|{x:#o}|{x:#x}|{x:#X}|{i:08d}|{x:_b}|{n:,d}|{v:.0f}|{v:.2f}|{v:.1%}|{v:.3E}|{v:.4G}|{v:,.2f}|{z:z.1f}",
                &values,
                &CancellationToken::default(),
            )?,
            "42      |   42   |******42|+42| 42|0b11111111|0o377|0xff|0XFF|00000042|1111_1111|12,345|1234|1234.50|123450.0%|1.234E+03|1234|1,234.50|0.0"
        );
        Ok(())
    }

    #[test]
    fn python_312_all_decimal_widths_and_precisions_use_the_pinned_index_classifier()
    -> Result<(), Box<dyn Error>> {
        let fixed = [
            "1",
            "1.2",
            "1.25",
            "1.250",
            "1.2500",
            "1.25000",
            "1.250000",
            "1.2500000",
            "1.25000000",
            "1.250000000",
        ];
        let prefixes = [
            "         ",
            "          ",
            "           ",
            "            ",
            "             ",
            "              ",
            "               ",
            "                ",
            "                 ",
            "                  ",
        ];
        for zero in PYTHON_UNICODE_15_DECIMAL_ZEROES {
            for decimal in 0..10u8 {
                let character = char::from_u32(zero + u32::from(decimal))
                    .ok_or("pinned decimal authority must contain a Unicode scalar")?;
                let cells = [
                    (
                        format!("{{a:>１{character}s}}"),
                        primitive(NativePrimitive::String("x".to_owned())),
                        format!("{}x", prefixes[usize::from(decimal)]),
                    ),
                    (
                        format!("{{a:.{character}s}}"),
                        primitive(NativePrimitive::String("abcdef".to_owned())),
                        "abcdef".chars().take(usize::from(decimal)).collect(),
                    ),
                    (
                        format!("{{a:.{character}f}}"),
                        primitive(NativePrimitive::Number(1.25)),
                        fixed[usize::from(decimal)].to_owned(),
                    ),
                ];
                for (template, value, expected) in cells {
                    assert_eq!(
                        NativeTextFormatter::format(
                            &template,
                            &BTreeMap::from([("a".to_owned(), value)]),
                            &CancellationToken::default()
                        )?,
                        expected,
                        "source decimal {character}"
                    );
                }
                let values = BTreeMap::from([
                    (
                        "a".to_owned(),
                        NativeValue::PreservedUnknown {
                            type_name: "JSON".to_owned(),
                            value: serde_json::json!({"text":"x"}),
                        },
                    ),
                    (
                        "b".to_owned(),
                        primitive(NativePrimitive::String(format!("１{character}"))),
                    ),
                ]);
                assert_eq!(
                    NativeTextFormatter::format(
                        "{a[text]:>{b}s}",
                        &values,
                        &CancellationToken::default()
                    )?,
                    format!("{}x", prefixes[usize::from(decimal)])
                );
            }
        }
        Ok(())
    }

    #[test]
    fn python_312_mixed_decimal_specs_preserve_flags_fill_conversion_and_nested_fields()
    -> Result<(), Box<dyn Error>> {
        for (template, value, expected) in [
            (
                "{a:>١２s}",
                NativePrimitive::String("x".to_owned()),
                "           x",
            ),
            ("{a:.٠３f}", NativePrimitive::Number(1.25), "1.250"),
            ("{a:١０,d}", NativePrimitive::Integer(1234), "     1,234"),
            ("{a:٠٥s}", NativePrimitive::String("x".to_owned()), "x    "),
            ("{a:٠٥d}", NativePrimitive::Integer(7), "    7"),
            ("{a:0٥d}", NativePrimitive::Integer(7), "00007"),
            (
                "{a:٥^٧s}",
                NativePrimitive::String("x".to_owned()),
                "٥٥٥x٥٥٥",
            ),
            (
                "{a:>٨.٣s}",
                NativePrimitive::String("abcdef".to_owned()),
                "     abc",
            ),
            (
                "{a!r:>٨.٣s}",
                NativePrimitive::String("x".to_owned()),
                "     'x'",
            ),
            (
                "{a!a:>٨s}",
                NativePrimitive::String("é".to_owned()),
                "  '\\xe9'",
            ),
            ("{a:>-١d}", NativePrimitive::Integer(7), "7"),
            ("{a:>s}", NativePrimitive::String("x".to_owned()), "x"),
            ("{a:.٠s}", NativePrimitive::String("x".to_owned()), ""),
            ("{a:>٠s}", NativePrimitive::String("x".to_owned()), "x"),
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    template,
                    &BTreeMap::from([("a".to_owned(), primitive(value))]),
                    &CancellationToken::default()
                )?,
                expected
            );
        }
        let values = BTreeMap::from([
            ("a".to_owned(), primitive(NativePrimitive::Number(1.25))),
            (
                "b".to_owned(),
                primitive(NativePrimitive::String("٨".to_owned())),
            ),
            (
                "c".to_owned(),
                primitive(NativePrimitive::String("٢".to_owned())),
            ),
        ]);
        assert_eq!(
            NativeTextFormatter::format("{a:>{b}.{c}f}", &values, &CancellationToken::default())?,
            "    1.25"
        );
        let zeroes = "٠".repeat(80);
        assert_eq!(
            NativeTextFormatter::format(
                &format!("{{a:.{zeroes}٣f}}"),
                &values,
                &CancellationToken::default()
            )?,
            "1.250"
        );
        Ok(())
    }

    #[test]
    fn python_312_decimal_spec_overflow_and_invalid_grammar_keep_source_error_order()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Number(1.25)))]);
        let too_large =
            NativeTextFormatError::InvalidTemplate("format number is too large".to_owned());
        let overflow = "٩".repeat(40);
        for spec in [
            format!(">{overflow}q"),
            format!(".{overflow}q"),
            format!(">{}q", isize::MAX as u128 + 1),
            format!(".{}q", isize::MAX as u128 + 1),
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a:{spec}}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(too_large.clone())
            );
        }
        for spec in [
            ".s",
            ".+٣f",
            ". ٣f",
            ".٣_٠f",
            ">١+٢s",
            ">²s",
            ".Ⅳf",
            ">\u{10d40}s",
            ">\u{1e5f1}s",
        ] {
            assert!(
                matches!(
                    NativeTextFormatter::format(
                        &format!("{{a:{spec}}}"),
                        &values,
                        &CancellationToken::default()
                    ),
                    Err(NativeTextFormatError::InvalidTemplate(_))
                ),
                "{spec}"
            );
        }
        assert_eq!(
            NativeTextFormatter::format(
                &format!("{{missing:>{overflow}q}}"),
                &values,
                &CancellationToken::default()
            ),
            Err(NativeTextFormatError::MissingField("missing".to_owned()))
        );
        assert_eq!(
            NativeTextFormatter::format(
                &format!("{{a!q:>{overflow}q}}"),
                &values,
                &CancellationToken::default()
            ),
            Err(NativeTextFormatError::InvalidTemplate(
                "conversion must be exactly !s, !r, or !a".to_owned()
            ))
        );
        let spec = FormatSpec::checked(&format!(">{}.{}q", isize::MAX, isize::MAX))?;
        assert_eq!(spec.width, Some(isize::MAX as usize));
        assert_eq!(spec.precision, Some(isize::MAX as usize));
        let values = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Integer(7)))]);
        assert_eq!(
            NativeTextFormatter::format("{a:.٣d}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::InvalidTemplate(
                "integer precision is unsupported".to_owned()
            ))
        );
        Ok(())
    }

    #[test]
    fn python_312_unicode_specs_preserve_native_byte_working_caps_cancel_and_recovery()
    -> Result<(), Box<dyn Error>> {
        let text = BTreeMap::from([(
            "a".to_owned(),
            primitive(NativePrimitive::String("abc".to_owned())),
        )]);
        assert_eq!(
            NativeTextFormatter::format("{a:.١٦٧٧٧٢١٧s}", &text, &CancellationToken::default())?,
            "abc"
        );
        assert_eq!(
            NativeTextFormatter::format("{a:>١٦٧٧٧٢١٧s}", &text, &CancellationToken::default()),
            Err(NativeTextFormatError::ResultTooLarge {
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES
            })
        );
        let numbers = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Number(1.25)))]);
        assert_eq!(
            NativeTextFormatter::format("{a:.١٦٧٧٧٢١٧f}", &numbers, &CancellationToken::default()),
            Err(NativeTextFormatError::ResultTooLarge {
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES
            })
        );
        assert_eq!(
            NativeTextFormatter::format("{a:.١٦٧٧٧٢١٧q}", &numbers, &CancellationToken::default()),
            Err(NativeTextFormatError::InvalidTemplate(
                "unsupported number format type".to_owned()
            ))
        );
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert_eq!(
            NativeTextFormatter::format("{a:.١٦٧٧٧٢١٧f}", &numbers, &cancellation),
            Err(NativeTextFormatError::Cancelled)
        );
        assert_eq!(
            NativeTextFormatter::format("{a:.٣f}", &numbers, &CancellationToken::default())?,
            "1.250"
        );
        Ok(())
    }

    #[test]
    fn python_312_string_zero_padding_keeps_source_align_fill_and_precision()
    -> Result<(), Box<dyn Error>> {
        for (template, value, expected) in [
            ("{a:0}", "x", "x"),
            ("{a:00}", "x", "x"),
            ("{a:05}", "x", "x0000"),
            ("{a:05s}", "x", "x0000"),
            ("{a:005s}", "x", "x0000"),
            ("{a:05.2s}", "x", "x0000"),
            ("{a:05.0s}", "x", "00000"),
            ("{a:0.2s}", "x", "x"),
            ("{a:<05s}", "x", "x0000"),
            ("{a:>05s}", "x", "0000x"),
            ("{a:^06s}", "x", "00x000"),
            ("{a:*<05s}", "x", "x****"),
            ("{a:*>05s}", "x", "****x"),
            ("{a:*^06s}", "x", "**x***"),
            ("{a: <05s}", "x", "x    "),
            ("{a: >05s}", "x", "    x"),
            ("{a:0<05s}", "x", "x0000"),
            ("{a:0>05s}", "x", "0000x"),
            ("{a:0^06s}", "x", "00x000"),
            ("{a:😀^06s}", "x", "😀😀x😀😀😀"),
            ("{a:١<0５.٢s}", "x", "x١١١١"),
            ("{a:٠٥s}", "x", "x    "),
            ("{a:0٥s}", "x", "x0000"),
            ("{a:0１٢.٣s}", "x", "x00000000000"),
            ("{a:>0８.٣s}", "x", "0000000x"),
            ("{a:^0７.٣s}", "x", "000x000"),
            ("{a:٥^0７.٣s}", "x", "٥٥٥x٥٥٥"),
            ("{a:*<00s}", "x", "x"),
            ("{a:0}", "abcdef", "abcdef"),
            ("{a:00}", "abcdef", "abcdef"),
            ("{a:05}", "abcdef", "abcdef"),
            ("{a:05s}", "abcdef", "abcdef"),
            ("{a:005s}", "abcdef", "abcdef"),
            ("{a:05.2s}", "abcdef", "ab000"),
            ("{a:05.0s}", "abcdef", "00000"),
            ("{a:0.2s}", "abcdef", "ab"),
            ("{a:<05s}", "abcdef", "abcdef"),
            ("{a:>05s}", "abcdef", "abcdef"),
            ("{a:^06s}", "abcdef", "abcdef"),
            ("{a:*<05s}", "abcdef", "abcdef"),
            ("{a:*>05s}", "abcdef", "abcdef"),
            ("{a:*^06s}", "abcdef", "abcdef"),
            ("{a: <05s}", "abcdef", "abcdef"),
            ("{a: >05s}", "abcdef", "abcdef"),
            ("{a:0<05s}", "abcdef", "abcdef"),
            ("{a:0>05s}", "abcdef", "abcdef"),
            ("{a:0^06s}", "abcdef", "abcdef"),
            ("{a:😀^06s}", "abcdef", "abcdef"),
            ("{a:١<0５.٢s}", "abcdef", "ab١١١"),
            ("{a:٠٥s}", "abcdef", "abcdef"),
            ("{a:0٥s}", "abcdef", "abcdef"),
            ("{a:0１٢.٣s}", "abcdef", "abc000000000"),
            ("{a:>0８.٣s}", "abcdef", "00000abc"),
            ("{a:^0７.٣s}", "abcdef", "00abc00"),
            ("{a:٥^0７.٣s}", "abcdef", "٥٥abc٥٥"),
            ("{a:*<00s}", "abcdef", "abcdef"),
            ("{a:0}", "é🙂", "é🙂"),
            ("{a:00}", "é🙂", "é🙂"),
            ("{a:05}", "é🙂", "é🙂000"),
            ("{a:05s}", "é🙂", "é🙂000"),
            ("{a:005s}", "é🙂", "é🙂000"),
            ("{a:05.2s}", "é🙂", "é🙂000"),
            ("{a:05.0s}", "é🙂", "00000"),
            ("{a:0.2s}", "é🙂", "é🙂"),
            ("{a:<05s}", "é🙂", "é🙂000"),
            ("{a:>05s}", "é🙂", "000é🙂"),
            ("{a:^06s}", "é🙂", "00é🙂00"),
            ("{a:*<05s}", "é🙂", "é🙂***"),
            ("{a:*>05s}", "é🙂", "***é🙂"),
            ("{a:*^06s}", "é🙂", "**é🙂**"),
            ("{a: <05s}", "é🙂", "é🙂   "),
            ("{a: >05s}", "é🙂", "   é🙂"),
            ("{a:0<05s}", "é🙂", "é🙂000"),
            ("{a:0>05s}", "é🙂", "000é🙂"),
            ("{a:0^06s}", "é🙂", "00é🙂00"),
            ("{a:😀^06s}", "é🙂", "😀😀é🙂😀😀"),
            ("{a:١<0５.٢s}", "é🙂", "é🙂١١١"),
            ("{a:٠٥s}", "é🙂", "é🙂   "),
            ("{a:0٥s}", "é🙂", "é🙂000"),
            ("{a:0１٢.٣s}", "é🙂", "é🙂0000000000"),
            ("{a:>0８.٣s}", "é🙂", "000000é🙂"),
            ("{a:^0７.٣s}", "é🙂", "00é🙂000"),
            ("{a:٥^0７.٣s}", "é🙂", "٥٥é🙂٥٥٥"),
            ("{a:*<00s}", "é🙂", "é🙂"),
            ("{a:0}", "", ""),
            ("{a:00}", "", ""),
            ("{a:05}", "", "00000"),
            ("{a:05s}", "", "00000"),
            ("{a:005s}", "", "00000"),
            ("{a:05.2s}", "", "00000"),
            ("{a:05.0s}", "", "00000"),
            ("{a:0.2s}", "", ""),
            ("{a:<05s}", "", "00000"),
            ("{a:>05s}", "", "00000"),
            ("{a:^06s}", "", "000000"),
            ("{a:*<05s}", "", "*****"),
            ("{a:*>05s}", "", "*****"),
            ("{a:*^06s}", "", "******"),
            ("{a: <05s}", "", "     "),
            ("{a: >05s}", "", "     "),
            ("{a:0<05s}", "", "00000"),
            ("{a:0>05s}", "", "00000"),
            ("{a:0^06s}", "", "000000"),
            ("{a:😀^06s}", "", "😀😀😀😀😀😀"),
            ("{a:١<0５.٢s}", "", "١١١١١"),
            ("{a:٠٥s}", "", "     "),
            ("{a:0٥s}", "", "00000"),
            ("{a:0１٢.٣s}", "", "000000000000"),
            ("{a:>0８.٣s}", "", "00000000"),
            ("{a:^0７.٣s}", "", "0000000"),
            ("{a:٥^0７.٣s}", "", "٥٥٥٥٥٥٥"),
            ("{a:*<00s}", "", ""),
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    template,
                    &BTreeMap::from([(
                        "a".to_owned(),
                        primitive(NativePrimitive::String(value.to_owned()))
                    )]),
                    &CancellationToken::default()
                )?,
                expected,
                "original source {template} for {value:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn python_312_string_zero_padding_composes_with_real_conversions_and_nested_values()
    -> Result<(), Box<dyn Error>> {
        for (template, value, expected) in [
            ("{a!s:05s}", primitive(NativePrimitive::Integer(7)), "70000"),
            (
                "{a!s:>08s}",
                primitive(NativePrimitive::Integer(7)),
                "00000007",
            ),
            (
                "{a!s:*^09.3s}",
                primitive(NativePrimitive::Integer(7)),
                "****7****",
            ),
            ("{a!s:0٥s}", primitive(NativePrimitive::Integer(7)), "70000"),
            (
                "{a!r:05s}",
                primitive(NativePrimitive::String("é".to_owned())),
                "'é'00",
            ),
            (
                "{a!r:>08s}",
                primitive(NativePrimitive::String("é".to_owned())),
                "00000'é'",
            ),
            (
                "{a!r:*^09.3s}",
                primitive(NativePrimitive::String("é".to_owned())),
                "***'é'***",
            ),
            (
                "{a!r:0٥s}",
                primitive(NativePrimitive::String("é".to_owned())),
                "'é'00",
            ),
            (
                "{a!a:05s}",
                primitive(NativePrimitive::String("é".to_owned())),
                "'\\xe9'",
            ),
            (
                "{a!a:>08s}",
                primitive(NativePrimitive::String("é".to_owned())),
                "00'\\xe9'",
            ),
            (
                "{a!a:*^09.3s}",
                primitive(NativePrimitive::String("é".to_owned())),
                "***'\\x***",
            ),
            (
                "{a!a:0٥s}",
                primitive(NativePrimitive::String("é".to_owned())),
                "'\\xe9'",
            ),
            ("{a!s:05s}", primitive(NativePrimitive::Null), "None0"),
            ("{a!s:>08s}", primitive(NativePrimitive::Null), "0000None"),
            (
                "{a!s:*^09.3s}",
                primitive(NativePrimitive::Null),
                "***Non***",
            ),
            ("{a!s:0٥s}", primitive(NativePrimitive::Null), "None0"),
            (
                "{a!r:05s}",
                NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value: serde_json::json!({"k":1}),
                },
                "{'k': 1}",
            ),
            (
                "{a!r:>08s}",
                NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value: serde_json::json!({"k":1}),
                },
                "{'k': 1}",
            ),
            (
                "{a!r:*^09.3s}",
                NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value: serde_json::json!({"k":1}),
                },
                "***{'k***",
            ),
            (
                "{a!r:0٥s}",
                NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value: serde_json::json!({"k":1}),
                },
                "{'k': 1}",
            ),
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    template,
                    &BTreeMap::from([("a".to_owned(), value)]),
                    &CancellationToken::default()
                )?,
                expected
            );
        }
        let values = BTreeMap::from([
            (
                "a".to_owned(),
                NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value: serde_json::json!({"text":"abcdef"}),
                },
            ),
            (
                "b".to_owned(),
                primitive(NativePrimitive::String("٨".to_owned())),
            ),
            (
                "c".to_owned(),
                primitive(NativePrimitive::String("٣".to_owned())),
            ),
        ]);
        assert_eq!(
            NativeTextFormatter::format(
                "{a[text]:0{b}.{c}s}",
                &values,
                &CancellationToken::default()
            )?,
            "abc00000"
        );
        assert_eq!(
            NativeTextFormatter::format(
                "{a[text]:*^0{b}.{c}s}",
                &values,
                &CancellationToken::default()
            )?,
            "**abc***"
        );
        let numbers = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Integer(-7)))]);
        assert_eq!(
            NativeTextFormatter::format(
                "{a:05}|{a:>05}|{a:٠٥}",
                &numbers,
                &CancellationToken::default()
            )?,
            "-0007|000-7|   -7"
        );
        Ok(())
    }

    #[test]
    fn python_312_string_zero_padding_rejects_source_invalid_flags_after_spec_parsing() {
        let values = BTreeMap::from([(
            "a".to_owned(),
            primitive(NativePrimitive::String("x".to_owned())),
        )]);
        let text_error =
            NativeTextFormatError::InvalidTemplate("numeric format applied to text".to_owned());
        for spec in [
            "=05s",
            "*=05s",
            "+05s",
            "-05s",
            " 05s",
            "#05s",
            "z05s",
            "05d",
            "05q",
            "+016777217s",
            "=016777217s",
            "016777217q",
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a:{spec}}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(text_error.clone()),
                "{spec}"
            );
        }
        for spec in ["05,s", "05_s"] {
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a:{spec}}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidTemplate(
                    "grouping is unsupported for this format type".to_owned()
                ))
            );
        }
        for spec in ["0.s", "0.+٢s", "+05.s"] {
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a:{spec}}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidTemplate(
                    "precision requires digits".to_owned()
                ))
            );
        }
        let overflow = "٩".repeat(40);
        for spec in [
            format!("+0{overflow}s"),
            format!("=0{overflow}s"),
            format!("0{overflow}q"),
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a:{spec}}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidTemplate(
                    "format number is too large".to_owned()
                ))
            );
        }
        assert_eq!(
            NativeTextFormatter::format("{missing:+05s}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::MissingField("missing".to_owned()))
        );
        assert_eq!(
            NativeTextFormatter::format(
                &format!("{{a!q:0{overflow}s}}"),
                &values,
                &CancellationToken::default()
            ),
            Err(NativeTextFormatError::InvalidTemplate(
                "conversion must be exactly !s, !r, or !a".to_owned()
            ))
        );
    }

    #[test]
    fn python_312_string_zero_padding_keeps_native_byte_caps_precancel_and_recovery()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([(
            "a".to_owned(),
            primitive(NativePrimitive::String("x".to_owned())),
        )]);
        assert_eq!(
            NativeTextFormatter::format("{a:0.16777217s}", &values, &CancellationToken::default())?,
            "x"
        );
        assert_eq!(
            NativeTextFormatter::format("{a:016777217s}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::ResultTooLarge {
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES
            })
        );
        let width = NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES / '😀'.len_utf8() + 2;
        assert_eq!(
            NativeTextFormatter::format(
                &format!("{{a:😀>0{width}s}}"),
                &values,
                &CancellationToken::default()
            ),
            Err(NativeTextFormatError::ResultTooLarge {
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES
            })
        );
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert_eq!(
            NativeTextFormatter::format("{a:016777217s}", &values, &cancellation),
            Err(NativeTextFormatError::Cancelled)
        );
        let zeroes = "٠".repeat(80);
        assert_eq!(
            NativeTextFormatter::format(
                &format!("{{a:0{zeroes}٥s}}"),
                &values,
                &CancellationToken::default()
            )?,
            "x0000"
        );
        Ok(())
    }

    #[test]
    fn python_312_integer_explicit_fill_wins_over_zero_with_sign_and_radix_prefix()
    -> Result<(), Box<dyn Error>> {
        for (template, value, expected) in [
            ("{a:05d}", primitive(NativePrimitive::Integer(7)), "00007"),
            ("{a:>05d}", primitive(NativePrimitive::Integer(7)), "00007"),
            ("{a:*<05d}", primitive(NativePrimitive::Integer(7)), "7****"),
            ("{a:*>05d}", primitive(NativePrimitive::Integer(7)), "****7"),
            (
                "{a:*^06d}",
                primitive(NativePrimitive::Integer(7)),
                "**7***",
            ),
            (
                "{a:*=+08d}",
                primitive(NativePrimitive::Integer(7)),
                "+******7",
            ),
            (
                "{a: =+08d}",
                primitive(NativePrimitive::Integer(7)),
                "+      7",
            ),
            (
                "{a:0=+08d}",
                primitive(NativePrimitive::Integer(7)),
                "+0000007",
            ),
            (
                "{a:😀=+08d}",
                primitive(NativePrimitive::Integer(7)),
                "+😀😀😀😀😀😀7",
            ),
            ("{a:٠٥d}", primitive(NativePrimitive::Integer(7)), "    7"),
            ("{a:0٥d}", primitive(NativePrimitive::Integer(7)), "00007"),
            (
                "{a:*<0１٢d}",
                primitive(NativePrimitive::Integer(7)),
                "7***********",
            ),
            (
                "{a:*>+#010x}",
                primitive(NativePrimitive::Integer(7)),
                "******+0x7",
            ),
            (
                "{a:*=+#010x}",
                primitive(NativePrimitive::Integer(7)),
                "+0x******7",
            ),
            (
                "{a:*=#010b}",
                primitive(NativePrimitive::Integer(7)),
                "0b*****111",
            ),
            (
                "{a:*=#010o}",
                primitive(NativePrimitive::Integer(7)),
                "0o*******7",
            ),
            (
                "{a:*>010,d}",
                primitive(NativePrimitive::Integer(7)),
                "*********7",
            ),
            (
                "{a:*^014_d}",
                primitive(NativePrimitive::Integer(7)),
                "******7*******",
            ),
            ("{a:05d}", primitive(NativePrimitive::Integer(-7)), "-0007"),
            ("{a:>05d}", primitive(NativePrimitive::Integer(-7)), "000-7"),
            (
                "{a:*<05d}",
                primitive(NativePrimitive::Integer(-7)),
                "-7***",
            ),
            (
                "{a:*>05d}",
                primitive(NativePrimitive::Integer(-7)),
                "***-7",
            ),
            (
                "{a:*^06d}",
                primitive(NativePrimitive::Integer(-7)),
                "**-7**",
            ),
            (
                "{a:*=+08d}",
                primitive(NativePrimitive::Integer(-7)),
                "-******7",
            ),
            (
                "{a: =+08d}",
                primitive(NativePrimitive::Integer(-7)),
                "-      7",
            ),
            (
                "{a:0=+08d}",
                primitive(NativePrimitive::Integer(-7)),
                "-0000007",
            ),
            (
                "{a:😀=+08d}",
                primitive(NativePrimitive::Integer(-7)),
                "-😀😀😀😀😀😀7",
            ),
            ("{a:٠٥d}", primitive(NativePrimitive::Integer(-7)), "   -7"),
            ("{a:0٥d}", primitive(NativePrimitive::Integer(-7)), "-0007"),
            (
                "{a:*<0１٢d}",
                primitive(NativePrimitive::Integer(-7)),
                "-7**********",
            ),
            (
                "{a:*>+#010x}",
                primitive(NativePrimitive::Integer(-7)),
                "******-0x7",
            ),
            (
                "{a:*=+#010x}",
                primitive(NativePrimitive::Integer(-7)),
                "-0x******7",
            ),
            (
                "{a:*=#010b}",
                primitive(NativePrimitive::Integer(-7)),
                "-0b****111",
            ),
            (
                "{a:*=#010o}",
                primitive(NativePrimitive::Integer(-7)),
                "-0o******7",
            ),
            (
                "{a:*>010,d}",
                primitive(NativePrimitive::Integer(-7)),
                "********-7",
            ),
            (
                "{a:*^014_d}",
                primitive(NativePrimitive::Integer(-7)),
                "******-7******",
            ),
            ("{a:05d}", primitive(NativePrimitive::Integer(0)), "00000"),
            ("{a:>05d}", primitive(NativePrimitive::Integer(0)), "00000"),
            ("{a:*<05d}", primitive(NativePrimitive::Integer(0)), "0****"),
            ("{a:*>05d}", primitive(NativePrimitive::Integer(0)), "****0"),
            (
                "{a:*^06d}",
                primitive(NativePrimitive::Integer(0)),
                "**0***",
            ),
            (
                "{a:*=+08d}",
                primitive(NativePrimitive::Integer(0)),
                "+******0",
            ),
            (
                "{a: =+08d}",
                primitive(NativePrimitive::Integer(0)),
                "+      0",
            ),
            (
                "{a:0=+08d}",
                primitive(NativePrimitive::Integer(0)),
                "+0000000",
            ),
            (
                "{a:😀=+08d}",
                primitive(NativePrimitive::Integer(0)),
                "+😀😀😀😀😀😀0",
            ),
            ("{a:٠٥d}", primitive(NativePrimitive::Integer(0)), "    0"),
            ("{a:0٥d}", primitive(NativePrimitive::Integer(0)), "00000"),
            (
                "{a:*<0１٢d}",
                primitive(NativePrimitive::Integer(0)),
                "0***********",
            ),
            (
                "{a:*>+#010x}",
                primitive(NativePrimitive::Integer(0)),
                "******+0x0",
            ),
            (
                "{a:*=+#010x}",
                primitive(NativePrimitive::Integer(0)),
                "+0x******0",
            ),
            (
                "{a:*=#010b}",
                primitive(NativePrimitive::Integer(0)),
                "0b*******0",
            ),
            (
                "{a:*=#010o}",
                primitive(NativePrimitive::Integer(0)),
                "0o*******0",
            ),
            (
                "{a:*>010,d}",
                primitive(NativePrimitive::Integer(0)),
                "*********0",
            ),
            (
                "{a:*^014_d}",
                primitive(NativePrimitive::Integer(0)),
                "******0*******",
            ),
            (
                "{a:05d}",
                primitive(NativePrimitive::Boolean(true)),
                "00001",
            ),
            (
                "{a:>05d}",
                primitive(NativePrimitive::Boolean(true)),
                "00001",
            ),
            (
                "{a:*<05d}",
                primitive(NativePrimitive::Boolean(true)),
                "1****",
            ),
            (
                "{a:*>05d}",
                primitive(NativePrimitive::Boolean(true)),
                "****1",
            ),
            (
                "{a:*^06d}",
                primitive(NativePrimitive::Boolean(true)),
                "**1***",
            ),
            (
                "{a:*=+08d}",
                primitive(NativePrimitive::Boolean(true)),
                "+******1",
            ),
            (
                "{a: =+08d}",
                primitive(NativePrimitive::Boolean(true)),
                "+      1",
            ),
            (
                "{a:0=+08d}",
                primitive(NativePrimitive::Boolean(true)),
                "+0000001",
            ),
            (
                "{a:😀=+08d}",
                primitive(NativePrimitive::Boolean(true)),
                "+😀😀😀😀😀😀1",
            ),
            (
                "{a:٠٥d}",
                primitive(NativePrimitive::Boolean(true)),
                "    1",
            ),
            (
                "{a:0٥d}",
                primitive(NativePrimitive::Boolean(true)),
                "00001",
            ),
            (
                "{a:*<0１٢d}",
                primitive(NativePrimitive::Boolean(true)),
                "1***********",
            ),
            (
                "{a:*>+#010x}",
                primitive(NativePrimitive::Boolean(true)),
                "******+0x1",
            ),
            (
                "{a:*=+#010x}",
                primitive(NativePrimitive::Boolean(true)),
                "+0x******1",
            ),
            (
                "{a:*=#010b}",
                primitive(NativePrimitive::Boolean(true)),
                "0b*******1",
            ),
            (
                "{a:*=#010o}",
                primitive(NativePrimitive::Boolean(true)),
                "0o*******1",
            ),
            (
                "{a:*>010,d}",
                primitive(NativePrimitive::Boolean(true)),
                "*********1",
            ),
            (
                "{a:*^014_d}",
                primitive(NativePrimitive::Boolean(true)),
                "******1*******",
            ),
            (
                "{a:05d}",
                primitive(NativePrimitive::Boolean(false)),
                "00000",
            ),
            (
                "{a:>05d}",
                primitive(NativePrimitive::Boolean(false)),
                "00000",
            ),
            (
                "{a:*<05d}",
                primitive(NativePrimitive::Boolean(false)),
                "0****",
            ),
            (
                "{a:*>05d}",
                primitive(NativePrimitive::Boolean(false)),
                "****0",
            ),
            (
                "{a:*^06d}",
                primitive(NativePrimitive::Boolean(false)),
                "**0***",
            ),
            (
                "{a:*=+08d}",
                primitive(NativePrimitive::Boolean(false)),
                "+******0",
            ),
            (
                "{a: =+08d}",
                primitive(NativePrimitive::Boolean(false)),
                "+      0",
            ),
            (
                "{a:0=+08d}",
                primitive(NativePrimitive::Boolean(false)),
                "+0000000",
            ),
            (
                "{a:😀=+08d}",
                primitive(NativePrimitive::Boolean(false)),
                "+😀😀😀😀😀😀0",
            ),
            (
                "{a:٠٥d}",
                primitive(NativePrimitive::Boolean(false)),
                "    0",
            ),
            (
                "{a:0٥d}",
                primitive(NativePrimitive::Boolean(false)),
                "00000",
            ),
            (
                "{a:*<0１٢d}",
                primitive(NativePrimitive::Boolean(false)),
                "0***********",
            ),
            (
                "{a:*>+#010x}",
                primitive(NativePrimitive::Boolean(false)),
                "******+0x0",
            ),
            (
                "{a:*=+#010x}",
                primitive(NativePrimitive::Boolean(false)),
                "+0x******0",
            ),
            (
                "{a:*=#010b}",
                primitive(NativePrimitive::Boolean(false)),
                "0b*******0",
            ),
            (
                "{a:*=#010o}",
                primitive(NativePrimitive::Boolean(false)),
                "0o*******0",
            ),
            (
                "{a:*>010,d}",
                primitive(NativePrimitive::Boolean(false)),
                "*********0",
            ),
            (
                "{a:*^014_d}",
                primitive(NativePrimitive::Boolean(false)),
                "******0*******",
            ),
            (
                "{a:05d}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "9223372036854775807",
            ),
            (
                "{a:>05d}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "9223372036854775807",
            ),
            (
                "{a:*<05d}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "9223372036854775807",
            ),
            (
                "{a:*>05d}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "9223372036854775807",
            ),
            (
                "{a:*^06d}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "9223372036854775807",
            ),
            (
                "{a:*=+08d}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "+9223372036854775807",
            ),
            (
                "{a: =+08d}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "+9223372036854775807",
            ),
            (
                "{a:0=+08d}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "+9223372036854775807",
            ),
            (
                "{a:😀=+08d}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "+9223372036854775807",
            ),
            (
                "{a:٠٥d}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "9223372036854775807",
            ),
            (
                "{a:0٥d}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "9223372036854775807",
            ),
            (
                "{a:*<0１٢d}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "9223372036854775807",
            ),
            (
                "{a:*>+#010x}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "+0x7fffffffffffffff",
            ),
            (
                "{a:*=+#010x}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "+0x7fffffffffffffff",
            ),
            (
                "{a:*=#010b}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "0b111111111111111111111111111111111111111111111111111111111111111",
            ),
            (
                "{a:*=#010o}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "0o777777777777777777777",
            ),
            (
                "{a:*>010,d}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "9,223,372,036,854,775,807",
            ),
            (
                "{a:*^014_d}",
                primitive(NativePrimitive::Integer(9223372036854775807)),
                "9_223_372_036_854_775_807",
            ),
            (
                "{a:05d}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-9223372036854775808",
            ),
            (
                "{a:>05d}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-9223372036854775808",
            ),
            (
                "{a:*<05d}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-9223372036854775808",
            ),
            (
                "{a:*>05d}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-9223372036854775808",
            ),
            (
                "{a:*^06d}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-9223372036854775808",
            ),
            (
                "{a:*=+08d}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-9223372036854775808",
            ),
            (
                "{a: =+08d}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-9223372036854775808",
            ),
            (
                "{a:0=+08d}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-9223372036854775808",
            ),
            (
                "{a:😀=+08d}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-9223372036854775808",
            ),
            (
                "{a:٠٥d}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-9223372036854775808",
            ),
            (
                "{a:0٥d}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-9223372036854775808",
            ),
            (
                "{a:*<0１٢d}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-9223372036854775808",
            ),
            (
                "{a:*>+#010x}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-0x8000000000000000",
            ),
            (
                "{a:*=+#010x}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-0x8000000000000000",
            ),
            (
                "{a:*=#010b}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-0b1000000000000000000000000000000000000000000000000000000000000000",
            ),
            (
                "{a:*=#010o}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-0o1000000000000000000000",
            ),
            (
                "{a:*>010,d}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-9,223,372,036,854,775,808",
            ),
            (
                "{a:*^014_d}",
                primitive(NativePrimitive::Integer(-9223372036854775808)),
                "-9_223_372_036_854_775_808",
            ),
            (
                "{a:05d}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "18446744073709551615",
            ),
            (
                "{a:>05d}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "18446744073709551615",
            ),
            (
                "{a:*<05d}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "18446744073709551615",
            ),
            (
                "{a:*>05d}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "18446744073709551615",
            ),
            (
                "{a:*^06d}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "18446744073709551615",
            ),
            (
                "{a:*=+08d}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "+18446744073709551615",
            ),
            (
                "{a: =+08d}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "+18446744073709551615",
            ),
            (
                "{a:0=+08d}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "+18446744073709551615",
            ),
            (
                "{a:😀=+08d}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "+18446744073709551615",
            ),
            (
                "{a:٠٥d}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "18446744073709551615",
            ),
            (
                "{a:0٥d}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "18446744073709551615",
            ),
            (
                "{a:*<0１٢d}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "18446744073709551615",
            ),
            (
                "{a:*>+#010x}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "+0xffffffffffffffff",
            ),
            (
                "{a:*=+#010x}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "+0xffffffffffffffff",
            ),
            (
                "{a:*=#010b}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "0b1111111111111111111111111111111111111111111111111111111111111111",
            ),
            (
                "{a:*=#010o}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "0o1777777777777777777777",
            ),
            (
                "{a:*>010,d}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "18,446,744,073,709,551,615",
            ),
            (
                "{a:*^014_d}",
                primitive(NativePrimitive::UnsignedInteger(18446744073709551615)),
                "18_446_744_073_709_551_615",
            ),
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    template,
                    &BTreeMap::from([("a".to_owned(), value)]),
                    &CancellationToken::default()
                )?,
                expected,
                "original Source {template}"
            );
        }
        Ok(())
    }

    #[test]
    fn python_312_floating_explicit_fill_wins_over_zero_with_precision_and_sign()
    -> Result<(), Box<dyn Error>> {
        for (template, value, expected) in [
            (
                "{a:08}",
                primitive(NativePrimitive::Number(1.25)),
                "00001.25",
            ),
            (
                "{a:>08}",
                primitive(NativePrimitive::Number(1.25)),
                "00001.25",
            ),
            (
                "{a:*<08}",
                primitive(NativePrimitive::Number(1.25)),
                "1.25****",
            ),
            (
                "{a:*>08}",
                primitive(NativePrimitive::Number(1.25)),
                "****1.25",
            ),
            (
                "{a:*^09}",
                primitive(NativePrimitive::Number(1.25)),
                "**1.25***",
            ),
            (
                "{a:*=+010.2f}",
                primitive(NativePrimitive::Number(1.25)),
                "+*****1.25",
            ),
            (
                "{a: =+010.2f}",
                primitive(NativePrimitive::Number(1.25)),
                "+     1.25",
            ),
            (
                "{a:0=+010.2f}",
                primitive(NativePrimitive::Number(1.25)),
                "+000001.25",
            ),
            (
                "{a:😀=+010.2f}",
                primitive(NativePrimitive::Number(1.25)),
                "+😀😀😀😀😀1.25",
            ),
            (
                "{a:*=#012.0e}",
                primitive(NativePrimitive::Number(1.25)),
                "******1.e+00",
            ),
            (
                "{a:*>012.3g}",
                primitive(NativePrimitive::Number(1.25)),
                "********1.25",
            ),
            (
                "{a:*^012.1%}",
                primitive(NativePrimitive::Number(1.25)),
                "***125.0%***",
            ),
            (
                "{a:*>014,.2f}",
                primitive(NativePrimitive::Number(1.25)),
                "**********1.25",
            ),
            (
                "{a:٠٨.٢f}",
                primitive(NativePrimitive::Number(1.25)),
                "    1.25",
            ),
            (
                "{a:0٨.٢f}",
                primitive(NativePrimitive::Number(1.25)),
                "00001.25",
            ),
            (
                "{a:*>z010.1f}",
                primitive(NativePrimitive::Number(1.25)),
                "*******1.2",
            ),
            (
                "{a:08}",
                primitive(NativePrimitive::Number(-1.25)),
                "-0001.25",
            ),
            (
                "{a:>08}",
                primitive(NativePrimitive::Number(-1.25)),
                "000-1.25",
            ),
            (
                "{a:*<08}",
                primitive(NativePrimitive::Number(-1.25)),
                "-1.25***",
            ),
            (
                "{a:*>08}",
                primitive(NativePrimitive::Number(-1.25)),
                "***-1.25",
            ),
            (
                "{a:*^09}",
                primitive(NativePrimitive::Number(-1.25)),
                "**-1.25**",
            ),
            (
                "{a:*=+010.2f}",
                primitive(NativePrimitive::Number(-1.25)),
                "-*****1.25",
            ),
            (
                "{a: =+010.2f}",
                primitive(NativePrimitive::Number(-1.25)),
                "-     1.25",
            ),
            (
                "{a:0=+010.2f}",
                primitive(NativePrimitive::Number(-1.25)),
                "-000001.25",
            ),
            (
                "{a:😀=+010.2f}",
                primitive(NativePrimitive::Number(-1.25)),
                "-😀😀😀😀😀1.25",
            ),
            (
                "{a:*=#012.0e}",
                primitive(NativePrimitive::Number(-1.25)),
                "-*****1.e+00",
            ),
            (
                "{a:*>012.3g}",
                primitive(NativePrimitive::Number(-1.25)),
                "*******-1.25",
            ),
            (
                "{a:*^012.1%}",
                primitive(NativePrimitive::Number(-1.25)),
                "**-125.0%***",
            ),
            (
                "{a:*>014,.2f}",
                primitive(NativePrimitive::Number(-1.25)),
                "*********-1.25",
            ),
            (
                "{a:٠٨.٢f}",
                primitive(NativePrimitive::Number(-1.25)),
                "   -1.25",
            ),
            (
                "{a:0٨.٢f}",
                primitive(NativePrimitive::Number(-1.25)),
                "-0001.25",
            ),
            (
                "{a:*>z010.1f}",
                primitive(NativePrimitive::Number(-1.25)),
                "******-1.2",
            ),
            (
                "{a:08}",
                primitive(NativePrimitive::Number(0.0)),
                "000000.0",
            ),
            (
                "{a:>08}",
                primitive(NativePrimitive::Number(0.0)),
                "000000.0",
            ),
            (
                "{a:*<08}",
                primitive(NativePrimitive::Number(0.0)),
                "0.0*****",
            ),
            (
                "{a:*>08}",
                primitive(NativePrimitive::Number(0.0)),
                "*****0.0",
            ),
            (
                "{a:*^09}",
                primitive(NativePrimitive::Number(0.0)),
                "***0.0***",
            ),
            (
                "{a:*=+010.2f}",
                primitive(NativePrimitive::Number(0.0)),
                "+*****0.00",
            ),
            (
                "{a: =+010.2f}",
                primitive(NativePrimitive::Number(0.0)),
                "+     0.00",
            ),
            (
                "{a:0=+010.2f}",
                primitive(NativePrimitive::Number(0.0)),
                "+000000.00",
            ),
            (
                "{a:😀=+010.2f}",
                primitive(NativePrimitive::Number(0.0)),
                "+😀😀😀😀😀0.00",
            ),
            (
                "{a:*=#012.0e}",
                primitive(NativePrimitive::Number(0.0)),
                "******0.e+00",
            ),
            (
                "{a:*>012.3g}",
                primitive(NativePrimitive::Number(0.0)),
                "***********0",
            ),
            (
                "{a:*^012.1%}",
                primitive(NativePrimitive::Number(0.0)),
                "****0.0%****",
            ),
            (
                "{a:*>014,.2f}",
                primitive(NativePrimitive::Number(0.0)),
                "**********0.00",
            ),
            (
                "{a:٠٨.٢f}",
                primitive(NativePrimitive::Number(0.0)),
                "    0.00",
            ),
            (
                "{a:0٨.٢f}",
                primitive(NativePrimitive::Number(0.0)),
                "00000.00",
            ),
            (
                "{a:*>z010.1f}",
                primitive(NativePrimitive::Number(0.0)),
                "*******0.0",
            ),
            (
                "{a:08}",
                primitive(NativePrimitive::Number(-0.0)),
                "-00000.0",
            ),
            (
                "{a:>08}",
                primitive(NativePrimitive::Number(-0.0)),
                "0000-0.0",
            ),
            (
                "{a:*<08}",
                primitive(NativePrimitive::Number(-0.0)),
                "-0.0****",
            ),
            (
                "{a:*>08}",
                primitive(NativePrimitive::Number(-0.0)),
                "****-0.0",
            ),
            (
                "{a:*^09}",
                primitive(NativePrimitive::Number(-0.0)),
                "**-0.0***",
            ),
            (
                "{a:*=+010.2f}",
                primitive(NativePrimitive::Number(-0.0)),
                "-*****0.00",
            ),
            (
                "{a: =+010.2f}",
                primitive(NativePrimitive::Number(-0.0)),
                "-     0.00",
            ),
            (
                "{a:0=+010.2f}",
                primitive(NativePrimitive::Number(-0.0)),
                "-000000.00",
            ),
            (
                "{a:😀=+010.2f}",
                primitive(NativePrimitive::Number(-0.0)),
                "-😀😀😀😀😀0.00",
            ),
            (
                "{a:*=#012.0e}",
                primitive(NativePrimitive::Number(-0.0)),
                "-*****0.e+00",
            ),
            (
                "{a:*>012.3g}",
                primitive(NativePrimitive::Number(-0.0)),
                "**********-0",
            ),
            (
                "{a:*^012.1%}",
                primitive(NativePrimitive::Number(-0.0)),
                "***-0.0%****",
            ),
            (
                "{a:*>014,.2f}",
                primitive(NativePrimitive::Number(-0.0)),
                "*********-0.00",
            ),
            (
                "{a:٠٨.٢f}",
                primitive(NativePrimitive::Number(-0.0)),
                "   -0.00",
            ),
            (
                "{a:0٨.٢f}",
                primitive(NativePrimitive::Number(-0.0)),
                "-0000.00",
            ),
            (
                "{a:*>z010.1f}",
                primitive(NativePrimitive::Number(-0.0)),
                "*******0.0",
            ),
            (
                "{a:08}",
                primitive(NativePrimitive::Number(1000000000000.0)),
                "1000000000000.0",
            ),
            (
                "{a:>08}",
                primitive(NativePrimitive::Number(1000000000000.0)),
                "1000000000000.0",
            ),
            (
                "{a:*<08}",
                primitive(NativePrimitive::Number(1000000000000.0)),
                "1000000000000.0",
            ),
            (
                "{a:*>08}",
                primitive(NativePrimitive::Number(1000000000000.0)),
                "1000000000000.0",
            ),
            (
                "{a:*^09}",
                primitive(NativePrimitive::Number(1000000000000.0)),
                "1000000000000.0",
            ),
            (
                "{a:*=+010.2f}",
                primitive(NativePrimitive::Number(1000000000000.0)),
                "+1000000000000.00",
            ),
            (
                "{a: =+010.2f}",
                primitive(NativePrimitive::Number(1000000000000.0)),
                "+1000000000000.00",
            ),
            (
                "{a:0=+010.2f}",
                primitive(NativePrimitive::Number(1000000000000.0)),
                "+1000000000000.00",
            ),
            (
                "{a:😀=+010.2f}",
                primitive(NativePrimitive::Number(1000000000000.0)),
                "+1000000000000.00",
            ),
            (
                "{a:*=#012.0e}",
                primitive(NativePrimitive::Number(1000000000000.0)),
                "******1.e+12",
            ),
            (
                "{a:*>012.3g}",
                primitive(NativePrimitive::Number(1000000000000.0)),
                "*******1e+12",
            ),
            (
                "{a:*^012.1%}",
                primitive(NativePrimitive::Number(1000000000000.0)),
                "100000000000000.0%",
            ),
            (
                "{a:*>014,.2f}",
                primitive(NativePrimitive::Number(1000000000000.0)),
                "1,000,000,000,000.00",
            ),
            (
                "{a:٠٨.٢f}",
                primitive(NativePrimitive::Number(1000000000000.0)),
                "1000000000000.00",
            ),
            (
                "{a:0٨.٢f}",
                primitive(NativePrimitive::Number(1000000000000.0)),
                "1000000000000.00",
            ),
            (
                "{a:*>z010.1f}",
                primitive(NativePrimitive::Number(1000000000000.0)),
                "1000000000000.0",
            ),
            (
                "{a:08}",
                primitive(NativePrimitive::Number(1e-07)),
                "0001e-07",
            ),
            (
                "{a:>08}",
                primitive(NativePrimitive::Number(1e-07)),
                "0001e-07",
            ),
            (
                "{a:*<08}",
                primitive(NativePrimitive::Number(1e-07)),
                "1e-07***",
            ),
            (
                "{a:*>08}",
                primitive(NativePrimitive::Number(1e-07)),
                "***1e-07",
            ),
            (
                "{a:*^09}",
                primitive(NativePrimitive::Number(1e-07)),
                "**1e-07**",
            ),
            (
                "{a:*=+010.2f}",
                primitive(NativePrimitive::Number(1e-07)),
                "+*****0.00",
            ),
            (
                "{a: =+010.2f}",
                primitive(NativePrimitive::Number(1e-07)),
                "+     0.00",
            ),
            (
                "{a:0=+010.2f}",
                primitive(NativePrimitive::Number(1e-07)),
                "+000000.00",
            ),
            (
                "{a:😀=+010.2f}",
                primitive(NativePrimitive::Number(1e-07)),
                "+😀😀😀😀😀0.00",
            ),
            (
                "{a:*=#012.0e}",
                primitive(NativePrimitive::Number(1e-07)),
                "******1.e-07",
            ),
            (
                "{a:*>012.3g}",
                primitive(NativePrimitive::Number(1e-07)),
                "*******1e-07",
            ),
            (
                "{a:*^012.1%}",
                primitive(NativePrimitive::Number(1e-07)),
                "****0.0%****",
            ),
            (
                "{a:*>014,.2f}",
                primitive(NativePrimitive::Number(1e-07)),
                "**********0.00",
            ),
            (
                "{a:٠٨.٢f}",
                primitive(NativePrimitive::Number(1e-07)),
                "    0.00",
            ),
            (
                "{a:0٨.٢f}",
                primitive(NativePrimitive::Number(1e-07)),
                "00000.00",
            ),
            (
                "{a:*>z010.1f}",
                primitive(NativePrimitive::Number(1e-07)),
                "*******0.0",
            ),
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    template,
                    &BTreeMap::from([("a".to_owned(), value)]),
                    &CancellationToken::default()
                )?,
                expected,
                "original Source {template}"
            );
        }
        Ok(())
    }

    #[test]
    fn python_312_numeric_fill_correction_preserves_string_conversion_and_nested_policies()
    -> Result<(), Box<dyn Error>> {
        for (template, value, expected) in [
            (
                "{a:*<05s}",
                primitive(NativePrimitive::String("x".to_owned())),
                "x****",
            ),
            (
                "{a:*>05s}",
                primitive(NativePrimitive::String("x".to_owned())),
                "****x",
            ),
            (
                "{a:*^06s}",
                primitive(NativePrimitive::String("x".to_owned())),
                "**x***",
            ),
            (
                "{a: <05s}",
                primitive(NativePrimitive::String("x".to_owned())),
                "x    ",
            ),
            (
                "{a: >05s}",
                primitive(NativePrimitive::String("x".to_owned())),
                "    x",
            ),
            (
                "{a:0<05s}",
                primitive(NativePrimitive::String("x".to_owned())),
                "x0000",
            ),
            (
                "{a:0>05s}",
                primitive(NativePrimitive::String("x".to_owned())),
                "0000x",
            ),
            (
                "{a:😀^06s}",
                primitive(NativePrimitive::String("x".to_owned())),
                "😀😀x😀😀😀",
            ),
            (
                "{a:٠٥s}",
                primitive(NativePrimitive::String("x".to_owned())),
                "x    ",
            ),
            (
                "{a:0٥s}",
                primitive(NativePrimitive::String("x".to_owned())),
                "x0000",
            ),
            (
                "{a:*<05s}",
                primitive(NativePrimitive::String("é🙂".to_owned())),
                "é🙂***",
            ),
            (
                "{a:*>05s}",
                primitive(NativePrimitive::String("é🙂".to_owned())),
                "***é🙂",
            ),
            (
                "{a:*^06s}",
                primitive(NativePrimitive::String("é🙂".to_owned())),
                "**é🙂**",
            ),
            (
                "{a: <05s}",
                primitive(NativePrimitive::String("é🙂".to_owned())),
                "é🙂   ",
            ),
            (
                "{a: >05s}",
                primitive(NativePrimitive::String("é🙂".to_owned())),
                "   é🙂",
            ),
            (
                "{a:0<05s}",
                primitive(NativePrimitive::String("é🙂".to_owned())),
                "é🙂000",
            ),
            (
                "{a:0>05s}",
                primitive(NativePrimitive::String("é🙂".to_owned())),
                "000é🙂",
            ),
            (
                "{a:😀^06s}",
                primitive(NativePrimitive::String("é🙂".to_owned())),
                "😀😀é🙂😀😀",
            ),
            (
                "{a:٠٥s}",
                primitive(NativePrimitive::String("é🙂".to_owned())),
                "é🙂   ",
            ),
            (
                "{a:0٥s}",
                primitive(NativePrimitive::String("é🙂".to_owned())),
                "é🙂000",
            ),
            (
                "{a:*<05s}",
                primitive(NativePrimitive::String("".to_owned())),
                "*****",
            ),
            (
                "{a:*>05s}",
                primitive(NativePrimitive::String("".to_owned())),
                "*****",
            ),
            (
                "{a:*^06s}",
                primitive(NativePrimitive::String("".to_owned())),
                "******",
            ),
            (
                "{a: <05s}",
                primitive(NativePrimitive::String("".to_owned())),
                "     ",
            ),
            (
                "{a: >05s}",
                primitive(NativePrimitive::String("".to_owned())),
                "     ",
            ),
            (
                "{a:0<05s}",
                primitive(NativePrimitive::String("".to_owned())),
                "00000",
            ),
            (
                "{a:0>05s}",
                primitive(NativePrimitive::String("".to_owned())),
                "00000",
            ),
            (
                "{a:😀^06s}",
                primitive(NativePrimitive::String("".to_owned())),
                "😀😀😀😀😀😀",
            ),
            (
                "{a:٠٥s}",
                primitive(NativePrimitive::String("".to_owned())),
                "     ",
            ),
            (
                "{a:0٥s}",
                primitive(NativePrimitive::String("".to_owned())),
                "00000",
            ),
            (
                "{a!s:*<08s}",
                primitive(NativePrimitive::Integer(-7)),
                "-7******",
            ),
            (
                "{a!s:*>08s}",
                primitive(NativePrimitive::Integer(-7)),
                "******-7",
            ),
            (
                "{a!s:*^09s}",
                primitive(NativePrimitive::Integer(-7)),
                "***-7****",
            ),
            (
                "{a!r:*<08s}",
                primitive(NativePrimitive::Number(1.25)),
                "1.25****",
            ),
            (
                "{a!r:*>08s}",
                primitive(NativePrimitive::Number(1.25)),
                "****1.25",
            ),
            (
                "{a!r:*^09s}",
                primitive(NativePrimitive::Number(1.25)),
                "**1.25***",
            ),
            (
                "{a!a:*<08s}",
                primitive(NativePrimitive::String("é".to_owned())),
                "'\\xe9'**",
            ),
            (
                "{a!a:*>08s}",
                primitive(NativePrimitive::String("é".to_owned())),
                "**'\\xe9'",
            ),
            (
                "{a!a:*^09s}",
                primitive(NativePrimitive::String("é".to_owned())),
                "*'\\xe9'**",
            ),
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    template,
                    &BTreeMap::from([("a".to_owned(), value)]),
                    &CancellationToken::default()
                )?,
                expected,
                "original Source {template}"
            );
        }
        let values = BTreeMap::from([
            ("a".to_owned(), primitive(NativePrimitive::Number(-1.25))),
            (
                "b".to_owned(),
                primitive(NativePrimitive::String("١０".to_owned())),
            ),
            (
                "c".to_owned(),
                primitive(NativePrimitive::String("٢".to_owned())),
            ),
        ]);
        assert_eq!(
            NativeTextFormatter::format(
                "{a:*=+0{b}.{c}f}",
                &values,
                &CancellationToken::default()
            )?,
            "-*****1.25"
        );
        let values = BTreeMap::from([
            (
                "a".to_owned(),
                NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value: serde_json::json!({"integer":-7}),
                },
            ),
            (
                "b".to_owned(),
                primitive(NativePrimitive::String("١０".to_owned())),
            ),
        ]);
        assert_eq!(
            NativeTextFormatter::format(
                "{a[integer]:*=+#0{b}x}",
                &values,
                &CancellationToken::default()
            )?,
            "-0x******7"
        );
        Ok(())
    }

    #[test]
    fn python_312_numeric_explicit_fill_preserves_error_phase_byte_caps_and_precancel()
    -> Result<(), Box<dyn Error>> {
        let integers = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Integer(7)))]);
        assert_eq!(
            NativeTextFormatter::format(
                "{missing:*=+08q}",
                &integers,
                &CancellationToken::default()
            ),
            Err(NativeTextFormatError::MissingField("missing".to_owned()))
        );
        assert_eq!(
            NativeTextFormatter::format("{a!q:*=+08q}", &integers, &CancellationToken::default()),
            Err(NativeTextFormatError::InvalidTemplate(
                "conversion must be exactly !s, !r, or !a".to_owned()
            ))
        );
        assert_eq!(
            NativeTextFormatter::format(
                "{a:*>016777217q}",
                &integers,
                &CancellationToken::default()
            ),
            Err(NativeTextFormatError::InvalidTemplate(
                "unsupported integer format type".to_owned()
            ))
        );
        let overflow = "٩".repeat(40);
        assert_eq!(
            NativeTextFormatter::format(
                &format!("{{a:*=+0{overflow}q}}"),
                &integers,
                &CancellationToken::default()
            ),
            Err(NativeTextFormatError::InvalidTemplate(
                "format number is too large".to_owned()
            ))
        );
        for template in ["{a:*>0１+٢d}", "{a:*>0²d}"] {
            assert!(matches!(
                NativeTextFormatter::format(template, &integers, &CancellationToken::default()),
                Err(NativeTextFormatError::InvalidTemplate(_))
            ));
        }
        let numbers = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Number(-1.25)))]);
        assert_eq!(
            NativeTextFormatter::format("{a:*=+08.+٢f}", &numbers, &CancellationToken::default()),
            Err(NativeTextFormatError::InvalidTemplate(
                "precision requires digits".to_owned()
            ))
        );
        assert_eq!(
            NativeTextFormatter::format(
                "{a:*=+016777217.2f}",
                &numbers,
                &CancellationToken::default()
            ),
            Err(NativeTextFormatError::ResultTooLarge {
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES
            })
        );
        let width = NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES / '😀'.len_utf8() + 8;
        let template = format!("{{a:😀=+0{width}.2f}}");
        assert_eq!(
            NativeTextFormatter::format(&template, &numbers, &CancellationToken::default()),
            Err(NativeTextFormatError::ResultTooLarge {
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES
            })
        );
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert_eq!(
            NativeTextFormatter::format(&template, &numbers, &cancellation),
            Err(NativeTextFormatError::Cancelled)
        );
        assert_eq!(
            NativeTextFormatter::format("{a:*=+010.2f}", &numbers, &CancellationToken::default())?,
            "-*****1.25"
        );
        Ok(())
    }

    #[test]
    fn python_312_integer_character_zero_padding_preserves_literal_sign_and_scalar_characters()
    -> Result<(), Box<dyn Error>> {
        for (template, value, expected) in [
            ("{a:c}", primitive(NativePrimitive::Integer(0)), "\u{0000}"),
            (
                "{a:05c}",
                primitive(NativePrimitive::Integer(0)),
                "0000\u{0000}",
            ),
            (
                "{a:>05c}",
                primitive(NativePrimitive::Integer(0)),
                "0000\u{0000}",
            ),
            (
                "{a:<05c}",
                primitive(NativePrimitive::Integer(0)),
                "\u{0000}0000",
            ),
            (
                "{a:^06c}",
                primitive(NativePrimitive::Integer(0)),
                "00\u{0000}000",
            ),
            (
                "{a:=05c}",
                primitive(NativePrimitive::Integer(0)),
                "0000\u{0000}",
            ),
            (
                "{a:*=05c}",
                primitive(NativePrimitive::Integer(0)),
                "****\u{0000}",
            ),
            (
                "{a:*<05c}",
                primitive(NativePrimitive::Integer(0)),
                "\u{0000}****",
            ),
            (
                "{a:*>05c}",
                primitive(NativePrimitive::Integer(0)),
                "****\u{0000}",
            ),
            (
                "{a:*^06c}",
                primitive(NativePrimitive::Integer(0)),
                "**\u{0000}***",
            ),
            (
                "{a: =05c}",
                primitive(NativePrimitive::Integer(0)),
                "    \u{0000}",
            ),
            (
                "{a:0=05c}",
                primitive(NativePrimitive::Integer(0)),
                "0000\u{0000}",
            ),
            (
                "{a:😀=05c}",
                primitive(NativePrimitive::Integer(0)),
                "😀😀😀😀\u{0000}",
            ),
            (
                "{a:٠٥c}",
                primitive(NativePrimitive::Integer(0)),
                "    \u{0000}",
            ),
            (
                "{a:0٥c}",
                primitive(NativePrimitive::Integer(0)),
                "0000\u{0000}",
            ),
            (
                "{a:*=0１٢c}",
                primitive(NativePrimitive::Integer(0)),
                "***********\u{0000}",
            ),
            ("{a:c}", primitive(NativePrimitive::Integer(1)), "\u{0001}"),
            (
                "{a:05c}",
                primitive(NativePrimitive::Integer(1)),
                "0000\u{0001}",
            ),
            (
                "{a:>05c}",
                primitive(NativePrimitive::Integer(1)),
                "0000\u{0001}",
            ),
            (
                "{a:<05c}",
                primitive(NativePrimitive::Integer(1)),
                "\u{0001}0000",
            ),
            (
                "{a:^06c}",
                primitive(NativePrimitive::Integer(1)),
                "00\u{0001}000",
            ),
            (
                "{a:=05c}",
                primitive(NativePrimitive::Integer(1)),
                "0000\u{0001}",
            ),
            (
                "{a:*=05c}",
                primitive(NativePrimitive::Integer(1)),
                "****\u{0001}",
            ),
            (
                "{a:*<05c}",
                primitive(NativePrimitive::Integer(1)),
                "\u{0001}****",
            ),
            (
                "{a:*>05c}",
                primitive(NativePrimitive::Integer(1)),
                "****\u{0001}",
            ),
            (
                "{a:*^06c}",
                primitive(NativePrimitive::Integer(1)),
                "**\u{0001}***",
            ),
            (
                "{a: =05c}",
                primitive(NativePrimitive::Integer(1)),
                "    \u{0001}",
            ),
            (
                "{a:0=05c}",
                primitive(NativePrimitive::Integer(1)),
                "0000\u{0001}",
            ),
            (
                "{a:😀=05c}",
                primitive(NativePrimitive::Integer(1)),
                "😀😀😀😀\u{0001}",
            ),
            (
                "{a:٠٥c}",
                primitive(NativePrimitive::Integer(1)),
                "    \u{0001}",
            ),
            (
                "{a:0٥c}",
                primitive(NativePrimitive::Integer(1)),
                "0000\u{0001}",
            ),
            (
                "{a:*=0１٢c}",
                primitive(NativePrimitive::Integer(1)),
                "***********\u{0001}",
            ),
            ("{a:c}", primitive(NativePrimitive::Integer(32)), " "),
            ("{a:05c}", primitive(NativePrimitive::Integer(32)), "0000 "),
            ("{a:>05c}", primitive(NativePrimitive::Integer(32)), "0000 "),
            ("{a:<05c}", primitive(NativePrimitive::Integer(32)), " 0000"),
            (
                "{a:^06c}",
                primitive(NativePrimitive::Integer(32)),
                "00 000",
            ),
            ("{a:=05c}", primitive(NativePrimitive::Integer(32)), "0000 "),
            (
                "{a:*=05c}",
                primitive(NativePrimitive::Integer(32)),
                "**** ",
            ),
            (
                "{a:*<05c}",
                primitive(NativePrimitive::Integer(32)),
                " ****",
            ),
            (
                "{a:*>05c}",
                primitive(NativePrimitive::Integer(32)),
                "**** ",
            ),
            (
                "{a:*^06c}",
                primitive(NativePrimitive::Integer(32)),
                "** ***",
            ),
            (
                "{a: =05c}",
                primitive(NativePrimitive::Integer(32)),
                "     ",
            ),
            (
                "{a:0=05c}",
                primitive(NativePrimitive::Integer(32)),
                "0000 ",
            ),
            (
                "{a:😀=05c}",
                primitive(NativePrimitive::Integer(32)),
                "😀😀😀😀 ",
            ),
            ("{a:٠٥c}", primitive(NativePrimitive::Integer(32)), "     "),
            ("{a:0٥c}", primitive(NativePrimitive::Integer(32)), "0000 "),
            (
                "{a:*=0１٢c}",
                primitive(NativePrimitive::Integer(32)),
                "*********** ",
            ),
            ("{a:c}", primitive(NativePrimitive::Integer(43)), "+"),
            ("{a:05c}", primitive(NativePrimitive::Integer(43)), "0000+"),
            ("{a:>05c}", primitive(NativePrimitive::Integer(43)), "0000+"),
            ("{a:<05c}", primitive(NativePrimitive::Integer(43)), "+0000"),
            (
                "{a:^06c}",
                primitive(NativePrimitive::Integer(43)),
                "00+000",
            ),
            ("{a:=05c}", primitive(NativePrimitive::Integer(43)), "0000+"),
            (
                "{a:*=05c}",
                primitive(NativePrimitive::Integer(43)),
                "****+",
            ),
            (
                "{a:*<05c}",
                primitive(NativePrimitive::Integer(43)),
                "+****",
            ),
            (
                "{a:*>05c}",
                primitive(NativePrimitive::Integer(43)),
                "****+",
            ),
            (
                "{a:*^06c}",
                primitive(NativePrimitive::Integer(43)),
                "**+***",
            ),
            (
                "{a: =05c}",
                primitive(NativePrimitive::Integer(43)),
                "    +",
            ),
            (
                "{a:0=05c}",
                primitive(NativePrimitive::Integer(43)),
                "0000+",
            ),
            (
                "{a:😀=05c}",
                primitive(NativePrimitive::Integer(43)),
                "😀😀😀😀+",
            ),
            ("{a:٠٥c}", primitive(NativePrimitive::Integer(43)), "    +"),
            ("{a:0٥c}", primitive(NativePrimitive::Integer(43)), "0000+"),
            (
                "{a:*=0１٢c}",
                primitive(NativePrimitive::Integer(43)),
                "***********+",
            ),
            ("{a:c}", primitive(NativePrimitive::Integer(45)), "-"),
            ("{a:05c}", primitive(NativePrimitive::Integer(45)), "0000-"),
            ("{a:>05c}", primitive(NativePrimitive::Integer(45)), "0000-"),
            ("{a:<05c}", primitive(NativePrimitive::Integer(45)), "-0000"),
            (
                "{a:^06c}",
                primitive(NativePrimitive::Integer(45)),
                "00-000",
            ),
            ("{a:=05c}", primitive(NativePrimitive::Integer(45)), "0000-"),
            (
                "{a:*=05c}",
                primitive(NativePrimitive::Integer(45)),
                "****-",
            ),
            (
                "{a:*<05c}",
                primitive(NativePrimitive::Integer(45)),
                "-****",
            ),
            (
                "{a:*>05c}",
                primitive(NativePrimitive::Integer(45)),
                "****-",
            ),
            (
                "{a:*^06c}",
                primitive(NativePrimitive::Integer(45)),
                "**-***",
            ),
            (
                "{a: =05c}",
                primitive(NativePrimitive::Integer(45)),
                "    -",
            ),
            (
                "{a:0=05c}",
                primitive(NativePrimitive::Integer(45)),
                "0000-",
            ),
            (
                "{a:😀=05c}",
                primitive(NativePrimitive::Integer(45)),
                "😀😀😀😀-",
            ),
            ("{a:٠٥c}", primitive(NativePrimitive::Integer(45)), "    -"),
            ("{a:0٥c}", primitive(NativePrimitive::Integer(45)), "0000-"),
            (
                "{a:*=0１٢c}",
                primitive(NativePrimitive::Integer(45)),
                "***********-",
            ),
            ("{a:c}", primitive(NativePrimitive::Integer(65)), "A"),
            ("{a:05c}", primitive(NativePrimitive::Integer(65)), "0000A"),
            ("{a:>05c}", primitive(NativePrimitive::Integer(65)), "0000A"),
            ("{a:<05c}", primitive(NativePrimitive::Integer(65)), "A0000"),
            (
                "{a:^06c}",
                primitive(NativePrimitive::Integer(65)),
                "00A000",
            ),
            ("{a:=05c}", primitive(NativePrimitive::Integer(65)), "0000A"),
            (
                "{a:*=05c}",
                primitive(NativePrimitive::Integer(65)),
                "****A",
            ),
            (
                "{a:*<05c}",
                primitive(NativePrimitive::Integer(65)),
                "A****",
            ),
            (
                "{a:*>05c}",
                primitive(NativePrimitive::Integer(65)),
                "****A",
            ),
            (
                "{a:*^06c}",
                primitive(NativePrimitive::Integer(65)),
                "**A***",
            ),
            (
                "{a: =05c}",
                primitive(NativePrimitive::Integer(65)),
                "    A",
            ),
            (
                "{a:0=05c}",
                primitive(NativePrimitive::Integer(65)),
                "0000A",
            ),
            (
                "{a:😀=05c}",
                primitive(NativePrimitive::Integer(65)),
                "😀😀😀😀A",
            ),
            ("{a:٠٥c}", primitive(NativePrimitive::Integer(65)), "    A"),
            ("{a:0٥c}", primitive(NativePrimitive::Integer(65)), "0000A"),
            (
                "{a:*=0１٢c}",
                primitive(NativePrimitive::Integer(65)),
                "***********A",
            ),
            ("{a:c}", primitive(NativePrimitive::Integer(233)), "é"),
            ("{a:05c}", primitive(NativePrimitive::Integer(233)), "0000é"),
            (
                "{a:>05c}",
                primitive(NativePrimitive::Integer(233)),
                "0000é",
            ),
            (
                "{a:<05c}",
                primitive(NativePrimitive::Integer(233)),
                "é0000",
            ),
            (
                "{a:^06c}",
                primitive(NativePrimitive::Integer(233)),
                "00é000",
            ),
            (
                "{a:=05c}",
                primitive(NativePrimitive::Integer(233)),
                "0000é",
            ),
            (
                "{a:*=05c}",
                primitive(NativePrimitive::Integer(233)),
                "****é",
            ),
            (
                "{a:*<05c}",
                primitive(NativePrimitive::Integer(233)),
                "é****",
            ),
            (
                "{a:*>05c}",
                primitive(NativePrimitive::Integer(233)),
                "****é",
            ),
            (
                "{a:*^06c}",
                primitive(NativePrimitive::Integer(233)),
                "**é***",
            ),
            (
                "{a: =05c}",
                primitive(NativePrimitive::Integer(233)),
                "    é",
            ),
            (
                "{a:0=05c}",
                primitive(NativePrimitive::Integer(233)),
                "0000é",
            ),
            (
                "{a:😀=05c}",
                primitive(NativePrimitive::Integer(233)),
                "😀😀😀😀é",
            ),
            ("{a:٠٥c}", primitive(NativePrimitive::Integer(233)), "    é"),
            ("{a:0٥c}", primitive(NativePrimitive::Integer(233)), "0000é"),
            (
                "{a:*=0１٢c}",
                primitive(NativePrimitive::Integer(233)),
                "***********é",
            ),
            ("{a:c}", primitive(NativePrimitive::Integer(65536)), "𐀀"),
            (
                "{a:05c}",
                primitive(NativePrimitive::Integer(65536)),
                "0000𐀀",
            ),
            (
                "{a:>05c}",
                primitive(NativePrimitive::Integer(65536)),
                "0000𐀀",
            ),
            (
                "{a:<05c}",
                primitive(NativePrimitive::Integer(65536)),
                "𐀀0000",
            ),
            (
                "{a:^06c}",
                primitive(NativePrimitive::Integer(65536)),
                "00𐀀000",
            ),
            (
                "{a:=05c}",
                primitive(NativePrimitive::Integer(65536)),
                "0000𐀀",
            ),
            (
                "{a:*=05c}",
                primitive(NativePrimitive::Integer(65536)),
                "****𐀀",
            ),
            (
                "{a:*<05c}",
                primitive(NativePrimitive::Integer(65536)),
                "𐀀****",
            ),
            (
                "{a:*>05c}",
                primitive(NativePrimitive::Integer(65536)),
                "****𐀀",
            ),
            (
                "{a:*^06c}",
                primitive(NativePrimitive::Integer(65536)),
                "**𐀀***",
            ),
            (
                "{a: =05c}",
                primitive(NativePrimitive::Integer(65536)),
                "    𐀀",
            ),
            (
                "{a:0=05c}",
                primitive(NativePrimitive::Integer(65536)),
                "0000𐀀",
            ),
            (
                "{a:😀=05c}",
                primitive(NativePrimitive::Integer(65536)),
                "😀😀😀😀𐀀",
            ),
            (
                "{a:٠٥c}",
                primitive(NativePrimitive::Integer(65536)),
                "    𐀀",
            ),
            (
                "{a:0٥c}",
                primitive(NativePrimitive::Integer(65536)),
                "0000𐀀",
            ),
            (
                "{a:*=0１٢c}",
                primitive(NativePrimitive::Integer(65536)),
                "***********𐀀",
            ),
            ("{a:c}", primitive(NativePrimitive::Integer(128578)), "🙂"),
            (
                "{a:05c}",
                primitive(NativePrimitive::Integer(128578)),
                "0000🙂",
            ),
            (
                "{a:>05c}",
                primitive(NativePrimitive::Integer(128578)),
                "0000🙂",
            ),
            (
                "{a:<05c}",
                primitive(NativePrimitive::Integer(128578)),
                "🙂0000",
            ),
            (
                "{a:^06c}",
                primitive(NativePrimitive::Integer(128578)),
                "00🙂000",
            ),
            (
                "{a:=05c}",
                primitive(NativePrimitive::Integer(128578)),
                "0000🙂",
            ),
            (
                "{a:*=05c}",
                primitive(NativePrimitive::Integer(128578)),
                "****🙂",
            ),
            (
                "{a:*<05c}",
                primitive(NativePrimitive::Integer(128578)),
                "🙂****",
            ),
            (
                "{a:*>05c}",
                primitive(NativePrimitive::Integer(128578)),
                "****🙂",
            ),
            (
                "{a:*^06c}",
                primitive(NativePrimitive::Integer(128578)),
                "**🙂***",
            ),
            (
                "{a: =05c}",
                primitive(NativePrimitive::Integer(128578)),
                "    🙂",
            ),
            (
                "{a:0=05c}",
                primitive(NativePrimitive::Integer(128578)),
                "0000🙂",
            ),
            (
                "{a:😀=05c}",
                primitive(NativePrimitive::Integer(128578)),
                "😀😀😀😀🙂",
            ),
            (
                "{a:٠٥c}",
                primitive(NativePrimitive::Integer(128578)),
                "    🙂",
            ),
            (
                "{a:0٥c}",
                primitive(NativePrimitive::Integer(128578)),
                "0000🙂",
            ),
            (
                "{a:*=0１٢c}",
                primitive(NativePrimitive::Integer(128578)),
                "***********🙂",
            ),
            ("{a:c}", primitive(NativePrimitive::Integer(1114111)), "􏿿"),
            (
                "{a:05c}",
                primitive(NativePrimitive::Integer(1114111)),
                "0000􏿿",
            ),
            (
                "{a:>05c}",
                primitive(NativePrimitive::Integer(1114111)),
                "0000􏿿",
            ),
            (
                "{a:<05c}",
                primitive(NativePrimitive::Integer(1114111)),
                "􏿿0000",
            ),
            (
                "{a:^06c}",
                primitive(NativePrimitive::Integer(1114111)),
                "00􏿿000",
            ),
            (
                "{a:=05c}",
                primitive(NativePrimitive::Integer(1114111)),
                "0000􏿿",
            ),
            (
                "{a:*=05c}",
                primitive(NativePrimitive::Integer(1114111)),
                "****􏿿",
            ),
            (
                "{a:*<05c}",
                primitive(NativePrimitive::Integer(1114111)),
                "􏿿****",
            ),
            (
                "{a:*>05c}",
                primitive(NativePrimitive::Integer(1114111)),
                "****􏿿",
            ),
            (
                "{a:*^06c}",
                primitive(NativePrimitive::Integer(1114111)),
                "**􏿿***",
            ),
            (
                "{a: =05c}",
                primitive(NativePrimitive::Integer(1114111)),
                "    􏿿",
            ),
            (
                "{a:0=05c}",
                primitive(NativePrimitive::Integer(1114111)),
                "0000􏿿",
            ),
            (
                "{a:😀=05c}",
                primitive(NativePrimitive::Integer(1114111)),
                "😀😀😀😀􏿿",
            ),
            (
                "{a:٠٥c}",
                primitive(NativePrimitive::Integer(1114111)),
                "    􏿿",
            ),
            (
                "{a:0٥c}",
                primitive(NativePrimitive::Integer(1114111)),
                "0000􏿿",
            ),
            (
                "{a:*=0１٢c}",
                primitive(NativePrimitive::Integer(1114111)),
                "***********􏿿",
            ),
            (
                "{a:c}",
                primitive(NativePrimitive::Boolean(true)),
                "\u{0001}",
            ),
            (
                "{a:05c}",
                primitive(NativePrimitive::Boolean(true)),
                "0000\u{0001}",
            ),
            (
                "{a:>05c}",
                primitive(NativePrimitive::Boolean(true)),
                "0000\u{0001}",
            ),
            (
                "{a:<05c}",
                primitive(NativePrimitive::Boolean(true)),
                "\u{0001}0000",
            ),
            (
                "{a:^06c}",
                primitive(NativePrimitive::Boolean(true)),
                "00\u{0001}000",
            ),
            (
                "{a:=05c}",
                primitive(NativePrimitive::Boolean(true)),
                "0000\u{0001}",
            ),
            (
                "{a:*=05c}",
                primitive(NativePrimitive::Boolean(true)),
                "****\u{0001}",
            ),
            (
                "{a:*<05c}",
                primitive(NativePrimitive::Boolean(true)),
                "\u{0001}****",
            ),
            (
                "{a:*>05c}",
                primitive(NativePrimitive::Boolean(true)),
                "****\u{0001}",
            ),
            (
                "{a:*^06c}",
                primitive(NativePrimitive::Boolean(true)),
                "**\u{0001}***",
            ),
            (
                "{a: =05c}",
                primitive(NativePrimitive::Boolean(true)),
                "    \u{0001}",
            ),
            (
                "{a:0=05c}",
                primitive(NativePrimitive::Boolean(true)),
                "0000\u{0001}",
            ),
            (
                "{a:😀=05c}",
                primitive(NativePrimitive::Boolean(true)),
                "😀😀😀😀\u{0001}",
            ),
            (
                "{a:٠٥c}",
                primitive(NativePrimitive::Boolean(true)),
                "    \u{0001}",
            ),
            (
                "{a:0٥c}",
                primitive(NativePrimitive::Boolean(true)),
                "0000\u{0001}",
            ),
            (
                "{a:*=0１٢c}",
                primitive(NativePrimitive::Boolean(true)),
                "***********\u{0001}",
            ),
            (
                "{a:c}",
                primitive(NativePrimitive::Boolean(false)),
                "\u{0000}",
            ),
            (
                "{a:05c}",
                primitive(NativePrimitive::Boolean(false)),
                "0000\u{0000}",
            ),
            (
                "{a:>05c}",
                primitive(NativePrimitive::Boolean(false)),
                "0000\u{0000}",
            ),
            (
                "{a:<05c}",
                primitive(NativePrimitive::Boolean(false)),
                "\u{0000}0000",
            ),
            (
                "{a:^06c}",
                primitive(NativePrimitive::Boolean(false)),
                "00\u{0000}000",
            ),
            (
                "{a:=05c}",
                primitive(NativePrimitive::Boolean(false)),
                "0000\u{0000}",
            ),
            (
                "{a:*=05c}",
                primitive(NativePrimitive::Boolean(false)),
                "****\u{0000}",
            ),
            (
                "{a:*<05c}",
                primitive(NativePrimitive::Boolean(false)),
                "\u{0000}****",
            ),
            (
                "{a:*>05c}",
                primitive(NativePrimitive::Boolean(false)),
                "****\u{0000}",
            ),
            (
                "{a:*^06c}",
                primitive(NativePrimitive::Boolean(false)),
                "**\u{0000}***",
            ),
            (
                "{a: =05c}",
                primitive(NativePrimitive::Boolean(false)),
                "    \u{0000}",
            ),
            (
                "{a:0=05c}",
                primitive(NativePrimitive::Boolean(false)),
                "0000\u{0000}",
            ),
            (
                "{a:😀=05c}",
                primitive(NativePrimitive::Boolean(false)),
                "😀😀😀😀\u{0000}",
            ),
            (
                "{a:٠٥c}",
                primitive(NativePrimitive::Boolean(false)),
                "    \u{0000}",
            ),
            (
                "{a:0٥c}",
                primitive(NativePrimitive::Boolean(false)),
                "0000\u{0000}",
            ),
            (
                "{a:*=0１٢c}",
                primitive(NativePrimitive::Boolean(false)),
                "***********\u{0000}",
            ),
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    template,
                    &BTreeMap::from([("a".to_owned(), value)]),
                    &CancellationToken::default()
                )?,
                expected,
                "original Source {template}"
            );
        }
        Ok(())
    }

    #[test]
    fn python_312_character_range_flags_and_precision_fail_before_padding() {
        let range_error = NativeTextFormatError::InvalidTemplate(
            "character code point is outside the Unicode scalar range".to_owned(),
        );
        for value in [
            NativePrimitive::Integer(-1),
            NativePrimitive::Integer(i64::MIN),
            NativePrimitive::Integer(0x110000),
            NativePrimitive::Integer(i64::MAX),
            NativePrimitive::UnsignedInteger(u64::MAX),
            NativePrimitive::Integer(0xd800),
            NativePrimitive::Integer(0xdfff),
        ] {
            let values = BTreeMap::from([("a".to_owned(), primitive(value))]);
            for template in ["{a:c}", "{a:05c}", "{a:*=05c}", "{a:016777217c}"] {
                assert_eq!(
                    NativeTextFormatter::format(template, &values, &CancellationToken::default()),
                    Err(range_error.clone())
                );
            }
        }
        let values = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Integer(-1)))]);
        for template in ["{a:+05c}", "{a:-05c}", "{a: 05c}", "{a:#05c}"] {
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default()),
                Err(NativeTextFormatError::InvalidTemplate(
                    "unsupported character format option".to_owned()
                ))
            );
        }
        for template in ["{a:05,c}", "{a:05_c}"] {
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default()),
                Err(NativeTextFormatError::InvalidTemplate(
                    "grouping is unsupported for this format type".to_owned()
                ))
            );
        }
        for template in ["{a:.٢c}", "{a:+05.2c}", "{a:*<05.٢c}"] {
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default()),
                Err(NativeTextFormatError::InvalidTemplate(
                    "integer precision is unsupported".to_owned()
                ))
            );
        }
        assert_eq!(
            NativeTextFormatter::format("{a:z05c}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::InvalidTemplate(
                "negative-zero coercion requires a floating-point value".to_owned()
            ))
        );
        assert_eq!(
            NativeTextFormatter::format("{missing:05c}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::MissingField("missing".to_owned()))
        );
        assert_eq!(
            NativeTextFormatter::format("{a!q:05c}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::InvalidTemplate(
                "conversion must be exactly !s, !r, or !a".to_owned()
            ))
        );
        let overflow = "٩".repeat(40);
        assert_eq!(
            NativeTextFormatter::format(
                &format!("{{a:0{overflow}c}}"),
                &values,
                &CancellationToken::default()
            ),
            Err(NativeTextFormatError::InvalidTemplate(
                "format number is too large".to_owned()
            ))
        );
        for value in [
            NativePrimitive::Number(1.25),
            NativePrimitive::String("A".to_owned()),
            NativePrimitive::Null,
        ] {
            assert!(matches!(
                NativeTextFormatter::format(
                    "{a:05c}",
                    &BTreeMap::from([("a".to_owned(), primitive(value))]),
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidTemplate(_))
            ));
        }
    }

    #[test]
    fn python_312_character_zero_padding_uses_existing_conversions_and_nested_json_values()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Integer(65)))]);
        for conversion in ['s', 'r', 'a'] {
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a!{conversion}:05s}}"),
                    &values,
                    &CancellationToken::default()
                )?,
                "65000"
            );
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a!{conversion}:*>05s}}"),
                    &values,
                    &CancellationToken::default()
                )?,
                "***65"
            );
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a!{conversion}:05c}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidTemplate(
                    "numeric format applied to text".to_owned()
                ))
            );
        }
        let values = BTreeMap::from([
            (
                "a".to_owned(),
                NativeValue::PreservedUnknown {
                    type_name: "JSON".to_owned(),
                    value: serde_json::json!({"char":45}),
                },
            ),
            (
                "b".to_owned(),
                primitive(NativePrimitive::String("٥".to_owned())),
            ),
        ]);
        assert_eq!(
            NativeTextFormatter::format(
                "{a[char]:*=0{b}c}",
                &values,
                &CancellationToken::default()
            )?,
            "****-"
        );
        let values = BTreeMap::from([(
            "a".to_owned(),
            NativeValue::List {
                values: vec![primitive(NativePrimitive::Integer(43))],
            },
        )]);
        assert_eq!(
            NativeTextFormatter::format("{a[0]:05c}", &values, &CancellationToken::default())?,
            "0000+"
        );
        Ok(())
    }

    #[test]
    fn python_312_character_zero_padding_keeps_utf8_caps_precancel_and_recovery()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Integer(45)))]);
        assert_eq!(
            NativeTextFormatter::format("{a:016777217c}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::ResultTooLarge {
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES
            })
        );
        let width = NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES / '😀'.len_utf8() + 2;
        let template = format!("{{a:😀=0{width}c}}");
        assert_eq!(
            NativeTextFormatter::format(&template, &values, &CancellationToken::default()),
            Err(NativeTextFormatError::ResultTooLarge {
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES
            })
        );
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert_eq!(
            NativeTextFormatter::format(&template, &values, &cancellation),
            Err(NativeTextFormatError::Cancelled)
        );
        let zeroes = "٠".repeat(80);
        assert_eq!(
            NativeTextFormatter::format(
                &format!("{{a:0{zeroes}٥c}}"),
                &values,
                &CancellationToken::default()
            )?,
            "0000-"
        );
        Ok(())
    }

    #[test]
    fn python_312_ungrouped_invalid_integer_presentation_precedes_precision_and_options() {
        let error =
            NativeTextFormatError::InvalidTemplate("unsupported integer format type".to_owned());
        for value in [
            NativePrimitive::Integer(7),
            NativePrimitive::Integer(-7),
            NativePrimitive::Boolean(true),
            NativePrimitive::Boolean(false),
            NativePrimitive::Integer(0),
            NativePrimitive::Integer(i64::MAX),
            NativePrimitive::Integer(i64::MIN),
            NativePrimitive::UnsignedInteger(u64::MAX),
        ] {
            let values = BTreeMap::from([("a".to_owned(), primitive(value))]);
            for spec in [
                ".2q", "+.2q", "#.2q", "z.2q", "*=+05.2q", ".٢q", ".0s", ".2a", ".2D", ".2N",
                ".2B", ".2O", ".2C", ".2j", ".2😀", "05q", "z05q", "+05q", "#05q",
            ] {
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{a:{spec}}}"),
                        &values,
                        &CancellationToken::default()
                    ),
                    Err(error.clone()),
                    "original Source invalid type {spec}"
                );
            }
        }
    }

    #[test]
    fn python_312_known_integer_presentations_keep_precision_rejection_and_admitted_behavior()
    -> Result<(), Box<dyn Error>> {
        for value in [
            NativePrimitive::Integer(7),
            NativePrimitive::Integer(-7),
            NativePrimitive::Boolean(true),
            NativePrimitive::Boolean(false),
            NativePrimitive::Integer(0),
            NativePrimitive::Integer(i64::MAX),
            NativePrimitive::Integer(i64::MIN),
            NativePrimitive::UnsignedInteger(u64::MAX),
        ] {
            let values = BTreeMap::from([("a".to_owned(), primitive(value))]);
            for spec in [
                ".2d", ".2n", ".2b", ".2o", ".2x", ".2X", ".2c", "z.2d", "+.2c", "#.2c", ".2٤",
            ] {
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{a:{spec}}}"),
                        &values,
                        &CancellationToken::default()
                    ),
                    Err(NativeTextFormatError::InvalidTemplate(
                        "integer precision is unsupported".to_owned()
                    ))
                );
            }
        }
        let values = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Integer(65)))]);
        for (template, expected) in [
            ("{a:d}", "65"),
            ("{a:n}", "65"),
            ("{a:b}", "1000001"),
            ("{a:o}", "101"),
            ("{a:x}", "41"),
            ("{a:X}", "41"),
            ("{a:c}", "A"),
        ] {
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default())?,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn python_312_integer_type_precedence_keeps_parser_nested_fields_and_conversions()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Integer(7)))]);
        for (template, error) in [
            (
                "{missing:.2q}".to_owned(),
                NativeTextFormatError::MissingField("missing".to_owned()),
            ),
            (
                "{a!q:.2q}".to_owned(),
                NativeTextFormatError::InvalidTemplate(
                    "conversion must be exactly !s, !r, or !a".to_owned(),
                ),
            ),
            (
                "{a:.{b}q}".to_owned(),
                NativeTextFormatError::MissingField("b".to_owned()),
            ),
            (
                "{a:.q}".to_owned(),
                NativeTextFormatError::InvalidTemplate("precision requires digits".to_owned()),
            ),
            (
                "{a:.+2q}".to_owned(),
                NativeTextFormatError::InvalidTemplate("precision requires digits".to_owned()),
            ),
            (
                "{a:.2qq}".to_owned(),
                NativeTextFormatError::InvalidTemplate("unsupported format spec".to_owned()),
            ),
        ] {
            assert_eq!(
                NativeTextFormatter::format(&template, &values, &CancellationToken::default()),
                Err(error)
            );
        }
        let overflow = "٩".repeat(40);
        for spec in [format!(".{overflow}q"), format!("0{overflow}.2q")] {
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a:{spec}}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidTemplate(
                    "format number is too large".to_owned()
                ))
            );
        }
        for conversion in ['s', 'r', 'a'] {
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a!{conversion}:.2s}}"),
                    &values,
                    &CancellationToken::default()
                )?,
                "7"
            );
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a!{conversion}:05s}}"),
                    &values,
                    &CancellationToken::default()
                )?,
                "70000"
            );
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a!{conversion}:.2q}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidTemplate(
                    "numeric format applied to text".to_owned()
                ))
            );
        }
        for json in [
            serde_json::json!({"int":7}),
            serde_json::json!({"int":u64::MAX}),
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    "{a[int]:.٢q}",
                    &BTreeMap::from([(
                        "a".to_owned(),
                        NativeValue::PreservedUnknown {
                            type_name: "JSON".to_owned(),
                            value: json
                        }
                    )]),
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidTemplate(
                    "unsupported integer format type".to_owned()
                ))
            );
        }
        Ok(())
    }

    #[test]
    fn python_312_integer_type_precedence_retains_pointer_capacity_precancel_and_recovery()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Integer(7)))]);
        assert_eq!(
            NativeTextFormatter::format("{a:.16777217q}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::InvalidTemplate(
                "unsupported integer format type".to_owned()
            ))
        );
        assert_eq!(
            NativeTextFormatter::format("{a:.16777217d}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::InvalidTemplate(
                "integer precision is unsupported".to_owned()
            ))
        );
        assert_eq!(
            NativeTextFormatter::format(
                &format!("{{a:.{}q}}", isize::MAX),
                &values,
                &CancellationToken::default()
            ),
            Err(NativeTextFormatError::InvalidTemplate(
                "unsupported integer format type".to_owned()
            ))
        );
        assert_eq!(
            NativeTextFormatter::format(
                &format!("{{a:.{}q}}", isize::MAX as u128 + 1),
                &values,
                &CancellationToken::default()
            ),
            Err(NativeTextFormatError::InvalidTemplate(
                "format number is too large".to_owned()
            ))
        );
        assert_eq!(
            NativeTextFormatter::format("{a:016777217d}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::ResultTooLarge {
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES
            })
        );
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert_eq!(
            NativeTextFormatter::format("{a:.16777217q}", &values, &cancellation),
            Err(NativeTextFormatError::Cancelled)
        );
        assert_eq!(
            NativeTextFormatter::format("{a:*>05d}", &values, &CancellationToken::default())?,
            "****7"
        );
        Ok(())
    }

    #[test]
    fn python_312_grouping_type_validation_precedes_atom_precision_and_options() {
        let error = NativeTextFormatError::InvalidTemplate(
            "grouping is unsupported for this format type".to_owned(),
        );
        for value in [
            NativePrimitive::Integer(12345),
            NativePrimitive::Integer(-12345),
            NativePrimitive::Boolean(true),
            NativePrimitive::Boolean(false),
            NativePrimitive::Number(1.25),
            NativePrimitive::Number(-12345.5),
            NativePrimitive::String("abcdef".to_owned()),
        ] {
            let values = BTreeMap::from([("a".to_owned(), primitive(value))]);
            for grouping in [',', '_'] {
                for kind in ['n', 'c', 's', 'q', 'D', 'C'] {
                    for precision in ["", ".٢"] {
                        assert_eq!(
                            NativeTextFormatter::format(
                                &format!("{{a:*>014{grouping}{precision}{kind}}}"),
                                &values,
                                &CancellationToken::default()
                            ),
                            Err(error.clone())
                        );
                    }
                }
            }
            for kind in ['b', 'o', 'x', 'X'] {
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{a:,.2{kind}}}"),
                        &values,
                        &CancellationToken::default()
                    ),
                    Err(error.clone())
                );
            }
        }
    }

    #[test]
    fn python_312_source_admitted_grouping_preserves_existing_numeric_emission()
    -> Result<(), Box<dyn Error>> {
        for (template, value, expected) in [
            (
                "{a:*>014,}",
                primitive(NativePrimitive::Integer(12345)),
                "********12,345",
            ),
            (
                "{a:*>014,d}",
                primitive(NativePrimitive::Integer(12345)),
                "********12,345",
            ),
            (
                "{a:*>014_}",
                primitive(NativePrimitive::Integer(12345)),
                "********12_345",
            ),
            (
                "{a:*>014_d}",
                primitive(NativePrimitive::Integer(12345)),
                "********12_345",
            ),
            (
                "{a:*>014_b}",
                primitive(NativePrimitive::Integer(12345)),
                "11_0000_0011_1001",
            ),
            (
                "{a:*>014_o}",
                primitive(NativePrimitive::Integer(12345)),
                "********3_0071",
            ),
            (
                "{a:*>014_x}",
                primitive(NativePrimitive::Integer(12345)),
                "**********3039",
            ),
            (
                "{a:*>014_X}",
                primitive(NativePrimitive::Integer(12345)),
                "**********3039",
            ),
            (
                "{a:*>014,}",
                primitive(NativePrimitive::Integer(-12345)),
                "*******-12,345",
            ),
            (
                "{a:*>014,d}",
                primitive(NativePrimitive::Integer(-12345)),
                "*******-12,345",
            ),
            (
                "{a:*>014_}",
                primitive(NativePrimitive::Integer(-12345)),
                "*******-12_345",
            ),
            (
                "{a:*>014_d}",
                primitive(NativePrimitive::Integer(-12345)),
                "*******-12_345",
            ),
            (
                "{a:*>014_b}",
                primitive(NativePrimitive::Integer(-12345)),
                "-11_0000_0011_1001",
            ),
            (
                "{a:*>014_o}",
                primitive(NativePrimitive::Integer(-12345)),
                "*******-3_0071",
            ),
            (
                "{a:*>014_x}",
                primitive(NativePrimitive::Integer(-12345)),
                "*********-3039",
            ),
            (
                "{a:*>014_X}",
                primitive(NativePrimitive::Integer(-12345)),
                "*********-3039",
            ),
            (
                "{a:*>014,}",
                primitive(NativePrimitive::Boolean(true)),
                "*************1",
            ),
            (
                "{a:*>014,d}",
                primitive(NativePrimitive::Boolean(true)),
                "*************1",
            ),
            (
                "{a:*>014_}",
                primitive(NativePrimitive::Boolean(true)),
                "*************1",
            ),
            (
                "{a:*>014_d}",
                primitive(NativePrimitive::Boolean(true)),
                "*************1",
            ),
            (
                "{a:*>014_b}",
                primitive(NativePrimitive::Boolean(true)),
                "*************1",
            ),
            (
                "{a:*>014_o}",
                primitive(NativePrimitive::Boolean(true)),
                "*************1",
            ),
            (
                "{a:*>014_x}",
                primitive(NativePrimitive::Boolean(true)),
                "*************1",
            ),
            (
                "{a:*>014_X}",
                primitive(NativePrimitive::Boolean(true)),
                "*************1",
            ),
            (
                "{a:*>014,}",
                primitive(NativePrimitive::Boolean(false)),
                "*************0",
            ),
            (
                "{a:*>014,d}",
                primitive(NativePrimitive::Boolean(false)),
                "*************0",
            ),
            (
                "{a:*>014_}",
                primitive(NativePrimitive::Boolean(false)),
                "*************0",
            ),
            (
                "{a:*>014_d}",
                primitive(NativePrimitive::Boolean(false)),
                "*************0",
            ),
            (
                "{a:*>014_b}",
                primitive(NativePrimitive::Boolean(false)),
                "*************0",
            ),
            (
                "{a:*>014_o}",
                primitive(NativePrimitive::Boolean(false)),
                "*************0",
            ),
            (
                "{a:*>014_x}",
                primitive(NativePrimitive::Boolean(false)),
                "*************0",
            ),
            (
                "{a:*>014_X}",
                primitive(NativePrimitive::Boolean(false)),
                "*************0",
            ),
            (
                "{a:*>014,}",
                primitive(NativePrimitive::Number(1.25)),
                "**********1.25",
            ),
            (
                "{a:*>014,.٢}",
                primitive(NativePrimitive::Number(1.25)),
                "***********1.2",
            ),
            (
                "{a:*>014,f}",
                primitive(NativePrimitive::Number(1.25)),
                "******1.250000",
            ),
            (
                "{a:*>014,.٢f}",
                primitive(NativePrimitive::Number(1.25)),
                "**********1.25",
            ),
            (
                "{a:*>014,F}",
                primitive(NativePrimitive::Number(1.25)),
                "******1.250000",
            ),
            (
                "{a:*>014,.٢F}",
                primitive(NativePrimitive::Number(1.25)),
                "**********1.25",
            ),
            (
                "{a:*>014,e}",
                primitive(NativePrimitive::Number(1.25)),
                "**1.250000e+00",
            ),
            (
                "{a:*>014,.٢e}",
                primitive(NativePrimitive::Number(1.25)),
                "******1.25e+00",
            ),
            (
                "{a:*>014,E}",
                primitive(NativePrimitive::Number(1.25)),
                "**1.250000E+00",
            ),
            (
                "{a:*>014,.٢E}",
                primitive(NativePrimitive::Number(1.25)),
                "******1.25E+00",
            ),
            (
                "{a:*>014,g}",
                primitive(NativePrimitive::Number(1.25)),
                "**********1.25",
            ),
            (
                "{a:*>014,.٢g}",
                primitive(NativePrimitive::Number(1.25)),
                "***********1.2",
            ),
            (
                "{a:*>014,G}",
                primitive(NativePrimitive::Number(1.25)),
                "**********1.25",
            ),
            (
                "{a:*>014,.٢G}",
                primitive(NativePrimitive::Number(1.25)),
                "***********1.2",
            ),
            (
                "{a:*>014,%}",
                primitive(NativePrimitive::Number(1.25)),
                "***125.000000%",
            ),
            (
                "{a:*>014,.٢%}",
                primitive(NativePrimitive::Number(1.25)),
                "*******125.00%",
            ),
            (
                "{a:*>014_}",
                primitive(NativePrimitive::Number(1.25)),
                "**********1.25",
            ),
            (
                "{a:*>014_.٢}",
                primitive(NativePrimitive::Number(1.25)),
                "***********1.2",
            ),
            (
                "{a:*>014_f}",
                primitive(NativePrimitive::Number(1.25)),
                "******1.250000",
            ),
            (
                "{a:*>014_.٢f}",
                primitive(NativePrimitive::Number(1.25)),
                "**********1.25",
            ),
            (
                "{a:*>014_F}",
                primitive(NativePrimitive::Number(1.25)),
                "******1.250000",
            ),
            (
                "{a:*>014_.٢F}",
                primitive(NativePrimitive::Number(1.25)),
                "**********1.25",
            ),
            (
                "{a:*>014_e}",
                primitive(NativePrimitive::Number(1.25)),
                "**1.250000e+00",
            ),
            (
                "{a:*>014_.٢e}",
                primitive(NativePrimitive::Number(1.25)),
                "******1.25e+00",
            ),
            (
                "{a:*>014_E}",
                primitive(NativePrimitive::Number(1.25)),
                "**1.250000E+00",
            ),
            (
                "{a:*>014_.٢E}",
                primitive(NativePrimitive::Number(1.25)),
                "******1.25E+00",
            ),
            (
                "{a:*>014_g}",
                primitive(NativePrimitive::Number(1.25)),
                "**********1.25",
            ),
            (
                "{a:*>014_.٢g}",
                primitive(NativePrimitive::Number(1.25)),
                "***********1.2",
            ),
            (
                "{a:*>014_G}",
                primitive(NativePrimitive::Number(1.25)),
                "**********1.25",
            ),
            (
                "{a:*>014_.٢G}",
                primitive(NativePrimitive::Number(1.25)),
                "***********1.2",
            ),
            (
                "{a:*>014_%}",
                primitive(NativePrimitive::Number(1.25)),
                "***125.000000%",
            ),
            (
                "{a:*>014_.٢%}",
                primitive(NativePrimitive::Number(1.25)),
                "*******125.00%",
            ),
            (
                "{a:*>014,}",
                primitive(NativePrimitive::Number(-12345.5)),
                "*****-12,345.5",
            ),
            (
                "{a:*>014,.٢}",
                primitive(NativePrimitive::Number(-12345.5)),
                "******-1.2e+04",
            ),
            (
                "{a:*>014,f}",
                primitive(NativePrimitive::Number(-12345.5)),
                "-12,345.500000",
            ),
            (
                "{a:*>014,.٢f}",
                primitive(NativePrimitive::Number(-12345.5)),
                "****-12,345.50",
            ),
            (
                "{a:*>014,F}",
                primitive(NativePrimitive::Number(-12345.5)),
                "-12,345.500000",
            ),
            (
                "{a:*>014,.٢F}",
                primitive(NativePrimitive::Number(-12345.5)),
                "****-12,345.50",
            ),
            (
                "{a:*>014,e}",
                primitive(NativePrimitive::Number(-12345.5)),
                "*-1.234550e+04",
            ),
            (
                "{a:*>014,.٢e}",
                primitive(NativePrimitive::Number(-12345.5)),
                "*****-1.23e+04",
            ),
            (
                "{a:*>014,E}",
                primitive(NativePrimitive::Number(-12345.5)),
                "*-1.234550E+04",
            ),
            (
                "{a:*>014,.٢E}",
                primitive(NativePrimitive::Number(-12345.5)),
                "*****-1.23E+04",
            ),
            (
                "{a:*>014,g}",
                primitive(NativePrimitive::Number(-12345.5)),
                "*****-12,345.5",
            ),
            (
                "{a:*>014,.٢g}",
                primitive(NativePrimitive::Number(-12345.5)),
                "******-1.2e+04",
            ),
            (
                "{a:*>014,G}",
                primitive(NativePrimitive::Number(-12345.5)),
                "*****-12,345.5",
            ),
            (
                "{a:*>014,.٢G}",
                primitive(NativePrimitive::Number(-12345.5)),
                "******-1.2E+04",
            ),
            (
                "{a:*>014,%}",
                primitive(NativePrimitive::Number(-12345.5)),
                "-1,234,550.000000%",
            ),
            (
                "{a:*>014,.٢%}",
                primitive(NativePrimitive::Number(-12345.5)),
                "-1,234,550.00%",
            ),
            (
                "{a:*>014_}",
                primitive(NativePrimitive::Number(-12345.5)),
                "*****-12_345.5",
            ),
            (
                "{a:*>014_.٢}",
                primitive(NativePrimitive::Number(-12345.5)),
                "******-1.2e+04",
            ),
            (
                "{a:*>014_f}",
                primitive(NativePrimitive::Number(-12345.5)),
                "-12_345.500000",
            ),
            (
                "{a:*>014_.٢f}",
                primitive(NativePrimitive::Number(-12345.5)),
                "****-12_345.50",
            ),
            (
                "{a:*>014_F}",
                primitive(NativePrimitive::Number(-12345.5)),
                "-12_345.500000",
            ),
            (
                "{a:*>014_.٢F}",
                primitive(NativePrimitive::Number(-12345.5)),
                "****-12_345.50",
            ),
            (
                "{a:*>014_e}",
                primitive(NativePrimitive::Number(-12345.5)),
                "*-1.234550e+04",
            ),
            (
                "{a:*>014_.٢e}",
                primitive(NativePrimitive::Number(-12345.5)),
                "*****-1.23e+04",
            ),
            (
                "{a:*>014_E}",
                primitive(NativePrimitive::Number(-12345.5)),
                "*-1.234550E+04",
            ),
            (
                "{a:*>014_.٢E}",
                primitive(NativePrimitive::Number(-12345.5)),
                "*****-1.23E+04",
            ),
            (
                "{a:*>014_g}",
                primitive(NativePrimitive::Number(-12345.5)),
                "*****-12_345.5",
            ),
            (
                "{a:*>014_.٢g}",
                primitive(NativePrimitive::Number(-12345.5)),
                "******-1.2e+04",
            ),
            (
                "{a:*>014_G}",
                primitive(NativePrimitive::Number(-12345.5)),
                "*****-12_345.5",
            ),
            (
                "{a:*>014_.٢G}",
                primitive(NativePrimitive::Number(-12345.5)),
                "******-1.2E+04",
            ),
            (
                "{a:*>014_%}",
                primitive(NativePrimitive::Number(-12345.5)),
                "-1_234_550.000000%",
            ),
            (
                "{a:*>014_.٢%}",
                primitive(NativePrimitive::Number(-12345.5)),
                "-1_234_550.00%",
            ),
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    template,
                    &BTreeMap::from([("a".to_owned(), value)]),
                    &CancellationToken::default()
                )?,
                expected,
                "original Source {template}"
            );
        }
        let values = BTreeMap::from([
            ("a".to_owned(), primitive(NativePrimitive::Integer(-12345))),
            (
                "b".to_owned(),
                primitive(NativePrimitive::String("١٤".to_owned())),
            ),
        ]);
        assert_eq!(
            NativeTextFormatter::format("{a:*>0{b},d}", &values, &CancellationToken::default())?,
            "*******-12,345"
        );
        let values = BTreeMap::from([(
            "a".to_owned(),
            NativeValue::PreservedUnknown {
                type_name: "JSON".to_owned(),
                value: serde_json::json!({"int":-12345}),
            },
        )]);
        assert_eq!(
            NativeTextFormatter::format(
                "{a[int]:*>014_d}",
                &values,
                &CancellationToken::default()
            )?,
            "*******-12_345"
        );
        Ok(())
    }

    #[test]
    fn python_312_combined_repeated_and_postprecision_separators_keep_parser_error_order() {
        let values = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Integer(12345)))]);
        let combined = NativeTextFormatError::InvalidTemplate(
            "cannot combine comma and underscore grouping".to_owned(),
        );
        for spec in [
            ",_d", "_,d", ",_", "_,", ",_.2d", "_,.2d", ",_.d", "_,.d", ",_.2q", "_,.2q", ",_,d",
            "_,_d",
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a:{spec}}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(combined.clone())
            );
        }
        for spec in [
            ",,d", "__d", ",.2_d", "_.2,d", ",.2,d", "_.2_d", ",١d", "_١d",
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a:{spec}}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidTemplate(
                    "unsupported format spec".to_owned()
                ))
            );
        }
        for spec in [",,", "__", ",.2_", "_.2,"] {
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a:{spec}}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(NativeTextFormatError::InvalidTemplate(
                    "grouping is unsupported for this format type".to_owned()
                ))
            );
        }
        let overflow = "٩".repeat(40);
        assert_eq!(
            NativeTextFormatter::format(
                &format!("{{a:,_.{overflow}q}}"),
                &values,
                &CancellationToken::default()
            ),
            Err(combined)
        );
        assert_eq!(
            NativeTextFormatter::format(
                &format!("{{a:,,.{overflow}q}}"),
                &values,
                &CancellationToken::default()
            ),
            Err(NativeTextFormatError::InvalidTemplate(
                "unsupported format spec".to_owned()
            ))
        );
        assert_eq!(
            NativeTextFormatter::format(
                &format!("{{a:,.{overflow}q}}"),
                &values,
                &CancellationToken::default()
            ),
            Err(NativeTextFormatError::InvalidTemplate(
                "format number is too large".to_owned()
            ))
        );
        assert_eq!(
            NativeTextFormatter::format("{a:,.q}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::InvalidTemplate(
                "precision requires digits".to_owned()
            ))
        );
        assert_eq!(
            NativeTextFormatter::format("{missing:,_d}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::MissingField("missing".to_owned()))
        );
        assert_eq!(
            NativeTextFormatter::format("{a!q:,_d}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::InvalidTemplate(
                "conversion must be exactly !s, !r, or !a".to_owned()
            ))
        );
        assert_eq!(
            NativeTextFormatter::format("{a:,.{b}q}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::MissingField("b".to_owned()))
        );
    }

    #[test]
    fn python_312_grouping_validation_keeps_conversions_capacity_precancel_and_recovery()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Integer(12345)))]);
        let error = NativeTextFormatError::InvalidTemplate(
            "grouping is unsupported for this format type".to_owned(),
        );
        assert_eq!(
            NativeTextFormatter::format(
                "{a:*>016777217,.16777217q}",
                &values,
                &CancellationToken::default()
            ),
            Err(error.clone())
        );
        for conversion in ['s', 'r', 'a'] {
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a!{conversion}:,.2s}}"),
                    &values,
                    &CancellationToken::default()
                ),
                Err(error.clone())
            );
        }
        let text = BTreeMap::from([(
            "a".to_owned(),
            primitive(NativePrimitive::String("x".to_owned())),
        )]);
        assert_eq!(
            NativeTextFormatter::format("{a:+,.2s}", &text, &CancellationToken::default()),
            Err(error)
        );
        assert_eq!(
            NativeTextFormatter::format("{a:*>014,}", &text, &CancellationToken::default()),
            Err(NativeTextFormatError::InvalidTemplate(
                "numeric format applied to text".to_owned()
            ))
        );
        assert_eq!(
            NativeTextFormatter::format(
                "{a:*>016777217,d}",
                &values,
                &CancellationToken::default()
            ),
            Err(NativeTextFormatError::ResultTooLarge {
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES
            })
        );
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert_eq!(
            NativeTextFormatter::format("{a:,_d}", &values, &cancellation),
            Err(NativeTextFormatError::Cancelled)
        );
        assert_eq!(
            NativeTextFormatter::format("{a:*>014,d}", &values, &CancellationToken::default())?,
            "********12,345"
        );
        Ok(())
    }

    #[test]
    fn python_312_untyped_float_precision_preserves_zero_exponents_and_decimal_suffixes()
    -> Result<(), Box<dyn Error>> {
        let specs = [
            ".0", ".1", ".2", ".3", ".6", "#.0", "#.1", "#.2", "#.3", "#.6",
        ];
        let cases = [
            (
                0x0000000000000000,
                [
                    "0e+00", "0e+00", "0.0", "0.0", "0.0", "0.e+00", "0.e+00", "0.0", "0.00",
                    "0.00000",
                ],
            ),
            (
                0x8000000000000000,
                [
                    "-0e+00", "-0e+00", "-0.0", "-0.0", "-0.0", "-0.e+00", "-0.e+00", "-0.0",
                    "-0.00", "-0.00000",
                ],
            ),
            (
                0x3ff0000000000000,
                [
                    "1e+00", "1e+00", "1.0", "1.0", "1.0", "1.e+00", "1.e+00", "1.0", "1.00",
                    "1.00000",
                ],
            ),
            (
                0xbff0000000000000,
                [
                    "-1e+00", "-1e+00", "-1.0", "-1.0", "-1.0", "-1.e+00", "-1.e+00", "-1.0",
                    "-1.00", "-1.00000",
                ],
            ),
            (
                0x4023fae147ae147b,
                [
                    "1e+01", "1e+01", "1e+01", "9.99", "9.99", "1.e+01", "1.e+01", "1.0e+01",
                    "9.99", "9.99000",
                ],
            ),
            (
                0x4023ff7ced916873,
                [
                    "1e+01", "1e+01", "1e+01", "10.0", "9.999", "1.e+01", "1.e+01", "1.0e+01",
                    "10.0", "9.99900",
                ],
            ),
            (
                0x4058ff5c28f5c28f,
                [
                    "1e+02", "1e+02", "1e+02", "1e+02", "99.99", "1.e+02", "1.e+02", "1.0e+02",
                    "1.00e+02", "99.9900",
                ],
            ),
            (
                0x4059000000000000,
                [
                    "1e+02", "1e+02", "1e+02", "1e+02", "100.0", "1.e+02", "1.e+02", "1.0e+02",
                    "1.00e+02", "100.000",
                ],
            ),
            (
                0x3f1a36e2eb1c432d,
                [
                    "0.0001",
                    "0.0001",
                    "0.0001",
                    "0.0001",
                    "0.0001",
                    "0.0001",
                    "0.0001",
                    "0.00010",
                    "0.000100",
                    "0.000100000",
                ],
            ),
            (
                0x3f1a36371ea531a8,
                [
                    "0.0001",
                    "0.0001",
                    "0.0001",
                    "0.0001",
                    "9.999e-05",
                    "0.0001",
                    "0.0001",
                    "0.00010",
                    "0.000100",
                    "9.99900e-05",
                ],
            ),
            (
                0x3ee4f8b588e368f1,
                [
                    "1e-05",
                    "1e-05",
                    "1e-05",
                    "1e-05",
                    "1e-05",
                    "1.e-05",
                    "1.e-05",
                    "1.0e-05",
                    "1.00e-05",
                    "1.00000e-05",
                ],
            ),
            (
                0x3ff4000000000000,
                [
                    "1e+00", "1e+00", "1.2", "1.25", "1.25", "1.e+00", "1.e+00", "1.2", "1.25",
                    "1.25000",
                ],
            ),
            (
                0x3ff3c083126e978d,
                [
                    "1e+00", "1e+00", "1.2", "1.23", "1.2345", "1.e+00", "1.e+00", "1.2", "1.23",
                    "1.23450",
                ],
            ),
            (
                0x4415af1d78b58c40,
                [
                    "1e+20",
                    "1e+20",
                    "1e+20",
                    "1e+20",
                    "1e+20",
                    "1.e+20",
                    "1.e+20",
                    "1.0e+20",
                    "1.00e+20",
                    "1.00000e+20",
                ],
            ),
            (
                0x0000000000000001,
                [
                    "5e-324",
                    "5e-324",
                    "4.9e-324",
                    "4.94e-324",
                    "4.94066e-324",
                    "5.e-324",
                    "5.e-324",
                    "4.9e-324",
                    "4.94e-324",
                    "4.94066e-324",
                ],
            ),
        ];
        for (bits, outputs) in cases {
            let values = BTreeMap::from([(
                "a".to_owned(),
                primitive(NativePrimitive::Number(f64::from_bits(bits))),
            )]);
            for (spec, expected) in specs.into_iter().zip(outputs) {
                assert_eq!(
                    NativeTextFormatter::format(
                        &format!("{{a:{spec}}}"),
                        &values,
                        &CancellationToken::default(),
                    )?,
                    expected,
                    "Python 3.12 source precision {spec} for bits {bits:016x}",
                );
            }
        }
        Ok(())
    }

    #[test]
    fn python_312_explicit_general_precision_keeps_integer_spelling() -> Result<(), Box<dyn Error>>
    {
        for (number, spec, expected) in [
            (0.0, ".0g", "0"),
            (0.0, "#.0g", "0."),
            (-0.0, ".2G", "-0"),
            (-0.0, "#.3G", "-0.00"),
            (1.0, ".2g", "1"),
            (1.0, "#.3g", "1.00"),
            (9.99, ".2g", "10"),
            (9.99, "#.2G", "10."),
            (100.0, ".3g", "100"),
            (100.0, "#.3G", "100."),
            (0.0001, ".1g", "0.0001"),
            (1e20, ".2G", "1E+20"),
        ] {
            assert_eq!(
                NativeTextFormatter::format(
                    &format!("{{a:{spec}}}"),
                    &BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Number(number)))]),
                    &CancellationToken::default(),
                )?,
                expected,
                "Python 3.12 explicit precision {spec} for {number}",
            );
        }
        Ok(())
    }

    #[test]
    fn python_312_untyped_precision_composes_with_sign_padding_grouping_and_nested_specs()
    -> Result<(), Box<dyn Error>> {
        for (template, number, expected) in [
            ("{a:>8.2}", 1.0, "     1.0"),
            ("{a:+08.2}", 1.0, "+00001.0"),
            ("{a:z.2}", -0.0, "0.0"),
            ("{a:.{b}}", 1.0, "1.0"),
            ("{a:,.6}", 12345.0, "12,345.0"),
        ] {
            let values = BTreeMap::from([
                ("a".to_owned(), primitive(NativePrimitive::Number(number))),
                ("b".to_owned(), primitive(NativePrimitive::Integer(2))),
            ]);
            assert_eq!(
                NativeTextFormatter::format(template, &values, &CancellationToken::default())?,
                expected,
            );
        }
        Ok(())
    }

    #[test]
    fn numeric_precision_capacity_cancellation_and_recovery_fail_closed()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([
            ("a".to_owned(), primitive(NativePrimitive::Number(1.0))),
            (
                "b".to_owned(),
                primitive(NativePrimitive::String("café".to_owned())),
            ),
            ("c".to_owned(), primitive(NativePrimitive::Integer(7))),
        ]);
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        let precision = NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES + 1;
        for kind in ["", "f", "e", "g", "%"] {
            let template = format!("{{a:.{precision}{kind}}}");
            assert_eq!(
                NativeTextFormatter::format(&template, &values, &CancellationToken::default()),
                Err(NativeTextFormatError::ResultTooLarge {
                    maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES,
                }),
            );
            assert_eq!(
                NativeTextFormatter::format(&template, &values, &cancellation),
                Err(NativeTextFormatError::Cancelled),
            );
        }
        assert!(matches!(
            NativeTextFormatter::format(
                &format!("{{a:.{precision}q}}"),
                &values,
                &CancellationToken::default(),
            ),
            Err(NativeTextFormatError::InvalidTemplate(_)),
        ));
        assert!(matches!(
            NativeTextFormatter::format("{c:.2}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::InvalidTemplate(_)),
        ));
        assert_eq!(
            NativeTextFormatter::format(
                &format!("{{b:.{precision}s}}"),
                &values,
                &CancellationToken::default(),
            )?,
            "café",
        );
        assert_eq!(
            NativeTextFormatter::format("{a:.2}", &values, &CancellationToken::default())?,
            "1.0",
        );
        Ok(())
    }

    #[test]
    fn python_repr_and_result_allocation_bounds_fail_closed_before_repeating()
    -> Result<(), Box<dyn Error>> {
        let values = BTreeMap::from([
            (
                "a".to_owned(),
                primitive(NativePrimitive::String("can't \u{0007} café".to_owned())),
            ),
            ("b".to_owned(), primitive(NativePrimitive::Integer(1))),
        ]);
        assert_eq!(
            NativeTextFormatter::format("{a!r}|{a!a}", &values, &CancellationToken::default(),)?,
            "\"can't \\x07 café\"|\"can't \\x07 caf\\xe9\""
        );
        assert_eq!(
            NativeTextFormatter::format("{b:😀>999999999}", &values, &CancellationToken::default(),),
            Err(NativeTextFormatError::ResultTooLarge {
                maximum_bytes: NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES,
            })
        );
        Ok(())
    }

    #[test]
    fn invalid_missing_and_cancelled_formats_fail_closed() {
        let values = BTreeMap::new();
        assert!(matches!(
            NativeTextFormatter::format("{a}", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::MissingField(_))
        ));
        assert!(matches!(
            NativeTextFormatter::format("{", &values, &CancellationToken::default()),
            Err(NativeTextFormatError::InvalidTemplate(_))
        ));
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert_eq!(
            NativeTextFormatter::format("text", &values, &cancellation),
            Err(NativeTextFormatError::Cancelled)
        );
    }

    #[derive(Serialize)]
    struct TextTransformValidationArtifact {
        validation_id: &'static str,
        scope: &'static str,
        source_path: &'static str,
        source_sha256: &'static str,
        fixture_sha256: String,
        environment: TextTransformValidationEnvironment,
        cases: Vec<serde_json::Value>,
        passed: usize,
        failed: usize,
        skipped: usize,
    }

    #[derive(Serialize)]
    struct TextTransformValidationEnvironment {
        operating_system: &'static str,
        architecture: &'static str,
        backend_identity: &'static str,
    }

    #[test]
    fn val_node_002_text_transform_source_oracle() -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let regex = NativeTextRegex::checked(r"(?P<word>a)(b)?", NativeTextRegexFlags::default())?;
        let cases = vec![
            serde_json::json!({
                "case_id": "regex-python-octal-and-unmatched-group",
                "actual": regex.replace("a", "\\0-\\123-\\08-\\g<0>-\\g<word>-\\2", 0, &cancellation)?,
                "expected": "\u{0000}-S-\u{0000}8-a-a-",
                "passed": true,
            }),
            serde_json::json!({
                "case_id": "regex-python-zero-width-duplicate-order",
                "actual": NativeTextRegex::checked(r"x*", NativeTextRegexFlags::default())?
                    .replace("abxd", "-", 0, &cancellation)?,
                "expected": "-a-b--d-",
                "passed": true,
            }),
            serde_json::json!({
                "case_id": "regex-invalid-reference-fails-closed",
                "passed": matches!(
                    regex.replace("a", r"\3", 0, &cancellation),
                    Err(crate::NativeTextRegexError::InvalidReplacement(_))
                ),
            }),
            serde_json::json!({
                "case_id": "format-python-numeric-mini-language",
                "actual": NativeTextFormatter::format(
                    "{a:=+8d}|{b:,d}|{c:c}|{d:.2e}|{e:.3g}|{f:.2}|{g:d}",
                    &BTreeMap::from([
                        ("a".to_owned(), primitive(NativePrimitive::Integer(42))),
                        ("b".to_owned(), primitive(NativePrimitive::Integer(12_345))),
                        ("c".to_owned(), primitive(NativePrimitive::Integer(65))),
                        ("d".to_owned(), primitive(NativePrimitive::Number(1.0))),
                        ("e".to_owned(), primitive(NativePrimitive::Number(12_345.0))),
                        ("f".to_owned(), primitive(NativePrimitive::Number(12.34))),
                        ("g".to_owned(), primitive(NativePrimitive::Boolean(true))),
                    ]),
                    &cancellation,
                )?,
                "expected": "+     42|12,345|A|1.00e+00|1.23e+04|1.2e+01|1",
                "passed": true,
            }),
            serde_json::json!({
                "case_id": "format-preallocation-bound",
                "passed": matches!(
                    NativeTextFormatter::format(
                        "{a:😀>999999999}",
                        &BTreeMap::from([("a".to_owned(), primitive(NativePrimitive::Integer(1)))]),
                        &cancellation,
                    ),
                    Err(NativeTextFormatError::ResultTooLarge { .. })
                ),
            }),
        ];
        assert!(
            cases.iter().all(|case| {
                case.get("passed") == Some(&serde_json::Value::Bool(true))
                    && case
                        .get("actual")
                        .zip(case.get("expected"))
                        .is_none_or(|(actual, expected)| actual == expected)
            }),
            "source-oracle mismatch: {cases:#?}"
        );
        let fixture_bytes = serde_json::to_vec(&cases)?;
        let fixture_sha256 = format!("{:x}", Sha256::digest(&fixture_bytes));
        let artifact = TextTransformValidationArtifact {
            validation_id: "VAL-NODE-002",
            scope: "native text transform source oracle",
            source_path: "projects/comfy/ComfyUI/comfy_extras/nodes_string.py",
            source_sha256: "bb01963178f28efc6e3a9578aecb89f53f8265e89c0518da92ec21c4955e85f1",
            fixture_sha256,
            environment: TextTransformValidationEnvironment {
                operating_system: std::env::consts::OS,
                architecture: std::env::consts::ARCH,
                backend_identity: "native-rust-cpu-source-oracle",
            },
            passed: cases.len(),
            failed: 0,
            skipped: 0,
            cases,
        };
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root is unavailable")?;
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .map(|target| {
                if target.is_absolute() {
                    target
                } else {
                    workspace_root.join(target)
                }
            })
            .unwrap_or_else(|| workspace_root.join("target"));
        let artifact_directory = target.join("sim-parity");
        fs::create_dir_all(&artifact_directory)?;
        let mut artifact_bytes = serde_json::to_vec_pretty(&artifact)?;
        artifact_bytes.push(b'\n');
        fs::write(artifact_directory.join("val-node-002.json"), artifact_bytes)?;
        Ok(())
    }
}
