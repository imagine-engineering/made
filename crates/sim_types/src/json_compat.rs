use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NonFiniteJsonKind {
    Nan,
    PositiveInfinity,
    NegativeInfinity,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NonFiniteJsonToken {
    pub byte_offset: usize,
    pub source_length: usize,
    pub kind: NonFiniteJsonKind,
}

pub fn normalize_json_non_finite(bytes: &[u8]) -> (Vec<u8>, Vec<NonFiniteJsonToken>) {
    let mut normalized = Vec::with_capacity(bytes.len());
    let mut tokens = Vec::new();
    let mut position = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    while position < bytes.len() {
        let byte = bytes[position];
        if in_string {
            normalized.push(byte);
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            position += 1;
            continue;
        }
        if byte == b'"' {
            in_string = true;
            normalized.push(byte);
            position += 1;
            continue;
        }
        let candidates = [
            (b"-Infinity".as_slice(), NonFiniteJsonKind::NegativeInfinity),
            (b"Infinity".as_slice(), NonFiniteJsonKind::PositiveInfinity),
            (b"NaN".as_slice(), NonFiniteJsonKind::Nan),
        ];
        let matched = candidates.into_iter().find(|(candidate, _)| {
            bytes.get(position..position.saturating_add(candidate.len())) == Some(*candidate)
                && token_boundary(bytes, position, candidate.len())
        });
        if let Some((candidate, kind)) = matched {
            normalized.extend_from_slice(b"null");
            tokens.push(NonFiniteJsonToken {
                byte_offset: position,
                source_length: candidate.len(),
                kind,
            });
            position += candidate.len();
        } else {
            normalized.push(byte);
            position += 1;
        }
    }
    (normalized, tokens)
}

fn token_boundary(bytes: &[u8], start: usize, length: usize) -> bool {
    let is_identifier = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    let before = start
        .checked_sub(1)
        .and_then(|index| bytes.get(index))
        .copied();
    let after = start
        .checked_add(length)
        .and_then(|index| bytes.get(index))
        .copied();
    !before.is_some_and(is_identifier) && !after.is_some_and(is_identifier)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_only_bare_non_finite_tokens() {
        let source = br#"{"nan":NaN,"positive":Infinity,"negative":-Infinity,"text":"NaN Infinity -Infinity","escaped":"\"NaN\""}"#;
        let (normalized, tokens) = normalize_json_non_finite(source);
        assert_eq!(
            normalized,
            br#"{"nan":null,"positive":null,"negative":null,"text":"NaN Infinity -Infinity","escaped":"\"NaN\""}"#
        );
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].kind, NonFiniteJsonKind::Nan);
        assert_eq!(tokens[1].kind, NonFiniteJsonKind::PositiveInfinity);
        assert_eq!(tokens[2].kind, NonFiniteJsonKind::NegativeInfinity);
        for token in tokens {
            assert_eq!(
                source.get(token.byte_offset..token.byte_offset + token.source_length),
                match token.kind {
                    NonFiniteJsonKind::Nan => Some(b"NaN".as_slice()),
                    NonFiniteJsonKind::PositiveInfinity => Some(b"Infinity".as_slice()),
                    NonFiniteJsonKind::NegativeInfinity => Some(b"-Infinity".as_slice()),
                }
            );
        }
    }

    #[test]
    fn identifier_fragments_are_not_accepted_as_non_finite_tokens() {
        let source = br#"{"first":NaNvalue,"second":valueInfinity,"third":_NaN}"#;
        let (normalized, tokens) = normalize_json_non_finite(source);
        assert_eq!(normalized, source);
        assert!(tokens.is_empty());
    }
}
