use fancy_regex::{Regex, RegexBuilder};
use sim_types::CancellationToken;
use thiserror::Error;

pub const NATIVE_TEXT_REGEX_MAX_PATTERN_BYTES: usize = 16 * 1024;
pub const NATIVE_TEXT_REGEX_MAX_INPUT_BYTES: usize = 1024 * 1024;
pub const NATIVE_TEXT_REGEX_MAX_MATCHES: usize = 100_000;
pub const NATIVE_TEXT_REGEX_MAX_CAPTURE_BYTES: usize = 16 * 1024 * 1024;
pub const NATIVE_TEXT_REGEX_MAX_RESULT_BYTES: usize = 16 * 1024 * 1024;
pub const NATIVE_TEXT_REGEX_BACKTRACK_LIMIT: usize = 1_000_000;
const NATIVE_TEXT_REGEX_DELEGATE_SIZE_LIMIT: usize = 2 * 1024 * 1024;
const NATIVE_TEXT_REGEX_DELEGATE_DFA_SIZE_LIMIT: usize = 2 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeTextRegexFlags {
    pub case_insensitive: bool,
    pub multi_line: bool,
    pub dot_matches_new_line: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeTextRegexCaptureRows {
    capture_count: usize,
    rows: Vec<Vec<Option<String>>>,
    captured_bytes: usize,
}

impl NativeTextRegexCaptureRows {
    pub fn capture_count(&self) -> usize {
        self.capture_count
    }

    pub fn rows(&self) -> &[Vec<Option<String>>] {
        &self.rows
    }

    pub fn captured_bytes(&self) -> usize {
        self.captured_bytes
    }

    pub fn into_rows(self) -> Vec<Vec<Option<String>>> {
        self.rows
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum NativeTextRegexError {
    #[error("regex pattern is {actual_bytes} bytes, above the {maximum_bytes}-byte limit")]
    PatternTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    #[error("regex input is {actual_bytes} bytes, above the {maximum_bytes}-byte limit")]
    InputTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    #[error("invalid regex pattern: {0}")]
    InvalidPattern(String),
    #[error("regex execution exceeded a bounded runtime limit: {0}")]
    ExecutionLimit(String),
    #[error("regex result exceeded the {maximum_matches}-match limit")]
    MatchLimit { maximum_matches: usize },
    #[error("regex captures exceeded the {maximum_bytes}-byte limit")]
    CaptureLimit { maximum_bytes: usize },
    #[error("invalid regex replacement template: {0}")]
    InvalidReplacement(String),
    #[error("regex replacement exceeded the {maximum_bytes}-byte limit")]
    ResultLimit { maximum_bytes: usize },
    #[error("regex execution was cancelled")]
    Cancelled,
}

#[derive(Clone, Copy)]
struct NativeTextRegexLimits {
    maximum_input_bytes: usize,
    maximum_matches: usize,
    maximum_capture_bytes: usize,
    backtrack_limit: usize,
}

impl Default for NativeTextRegexLimits {
    fn default() -> Self {
        Self {
            maximum_input_bytes: NATIVE_TEXT_REGEX_MAX_INPUT_BYTES,
            maximum_matches: NATIVE_TEXT_REGEX_MAX_MATCHES,
            maximum_capture_bytes: NATIVE_TEXT_REGEX_MAX_CAPTURE_BYTES,
            backtrack_limit: NATIVE_TEXT_REGEX_BACKTRACK_LIMIT,
        }
    }
}

#[derive(Clone)]
pub struct NativeTextRegex {
    regex: Regex,
    limits: NativeTextRegexLimits,
}

impl NativeTextRegex {
    pub fn checked(
        pattern: &str,
        flags: NativeTextRegexFlags,
    ) -> Result<Self, NativeTextRegexError> {
        Self::checked_with_limits(pattern, flags, NativeTextRegexLimits::default())
    }

    fn checked_with_limits(
        pattern: &str,
        flags: NativeTextRegexFlags,
        limits: NativeTextRegexLimits,
    ) -> Result<Self, NativeTextRegexError> {
        if pattern.len() > NATIVE_TEXT_REGEX_MAX_PATTERN_BYTES {
            return Err(NativeTextRegexError::PatternTooLarge {
                actual_bytes: pattern.len(),
                maximum_bytes: NATIVE_TEXT_REGEX_MAX_PATTERN_BYTES,
            });
        }
        let mut builder = RegexBuilder::new(pattern);
        builder
            .case_insensitive(flags.case_insensitive)
            .multi_line(flags.multi_line)
            .dot_matches_new_line(flags.dot_matches_new_line)
            .backtrack_limit(limits.backtrack_limit)
            .delegate_size_limit(NATIVE_TEXT_REGEX_DELEGATE_SIZE_LIMIT)
            .delegate_dfa_size_limit(NATIVE_TEXT_REGEX_DELEGATE_DFA_SIZE_LIMIT);
        let regex = builder
            .build()
            .map_err(|error| NativeTextRegexError::InvalidPattern(error.to_string()))?;
        Ok(Self { regex, limits })
    }

    pub fn capture_count(&self) -> usize {
        self.regex.captures_len()
    }

    pub fn is_match(
        &self,
        input: &str,
        cancellation: &CancellationToken,
    ) -> Result<bool, NativeTextRegexError> {
        self.validate_input(input, cancellation)?;
        let result = self
            .regex
            .is_match(input)
            .map_err(|error| NativeTextRegexError::ExecutionLimit(error.to_string()))?;
        self.check_cancellation(cancellation)?;
        Ok(result)
    }

    pub fn capture_rows(
        &self,
        input: &str,
        cancellation: &CancellationToken,
    ) -> Result<NativeTextRegexCaptureRows, NativeTextRegexError> {
        self.validate_input(input, cancellation)?;
        let capture_count = self.regex.captures_len();
        let mut rows = Vec::new();
        let mut captured_bytes = 0usize;
        for captures in self.regex.captures_iter(input) {
            self.check_cancellation(cancellation)?;
            if rows.len() >= self.limits.maximum_matches {
                return Err(NativeTextRegexError::MatchLimit {
                    maximum_matches: self.limits.maximum_matches,
                });
            }
            let captures = captures
                .map_err(|error| NativeTextRegexError::ExecutionLimit(error.to_string()))?;
            let mut row = Vec::with_capacity(capture_count);
            for index in 0..capture_count {
                let value = captures.get(index).map(|matched| matched.as_str());
                if let Some(value) = value {
                    captured_bytes = captured_bytes.checked_add(value.len()).ok_or(
                        NativeTextRegexError::CaptureLimit {
                            maximum_bytes: self.limits.maximum_capture_bytes,
                        },
                    )?;
                    if captured_bytes > self.limits.maximum_capture_bytes {
                        return Err(NativeTextRegexError::CaptureLimit {
                            maximum_bytes: self.limits.maximum_capture_bytes,
                        });
                    }
                }
                row.push(value.map(str::to_owned));
            }
            rows.push(row);
        }
        self.check_cancellation(cancellation)?;
        Ok(NativeTextRegexCaptureRows {
            capture_count,
            rows,
            captured_bytes,
        })
    }

    pub fn replace(
        &self,
        input: &str,
        replacement: &str,
        count: usize,
        cancellation: &CancellationToken,
    ) -> Result<String, NativeTextRegexError> {
        self.validate_input(input, cancellation)?;
        let replacement = NativeTextRegexReplacement::checked(replacement, &self.regex)?;
        let mut output = String::with_capacity(input.len().min(NATIVE_TEXT_REGEX_MAX_RESULT_BYTES));
        let mut previous_end = 0usize;
        let mut search_position = 0usize;
        let mut replacement_count = 0usize;
        while search_position <= input.len() {
            self.check_cancellation(cancellation)?;
            if count != 0 && replacement_count >= count {
                break;
            }
            if replacement_count >= self.limits.maximum_matches {
                return Err(NativeTextRegexError::MatchLimit {
                    maximum_matches: self.limits.maximum_matches,
                });
            }
            let captures = self
                .regex
                .captures_from_pos(input, search_position)
                .map_err(|error| NativeTextRegexError::ExecutionLimit(error.to_string()))?;
            let Some(captures) = captures else {
                break;
            };
            let matched = captures.get(0).ok_or_else(|| {
                NativeTextRegexError::ExecutionLimit(
                    "regex capture row did not contain the whole match".to_owned(),
                )
            })?;
            append_bounded(
                &mut output,
                input.get(previous_end..matched.start()).ok_or_else(|| {
                    NativeTextRegexError::ExecutionLimit(
                        "regex match boundary was not valid UTF-8".to_owned(),
                    )
                })?,
            )?;
            replacement.append_expansion(&captures, &mut output)?;
            previous_end = matched.end();
            replacement_count += 1;
            if matched.start() == matched.end() {
                let Some(character) = input
                    .get(matched.end()..)
                    .and_then(|value| value.chars().next())
                else {
                    break;
                };
                search_position = matched.end().saturating_add(character.len_utf8());
            } else {
                search_position = matched.end();
            }
        }
        append_bounded(
            &mut output,
            input.get(previous_end..).ok_or_else(|| {
                NativeTextRegexError::ExecutionLimit(
                    "regex match boundary was not valid UTF-8".to_owned(),
                )
            })?,
        )?;
        self.check_cancellation(cancellation)?;
        Ok(output)
    }

    fn validate_input(
        &self,
        input: &str,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeTextRegexError> {
        self.check_cancellation(cancellation)?;
        if input.len() > self.limits.maximum_input_bytes {
            return Err(NativeTextRegexError::InputTooLarge {
                actual_bytes: input.len(),
                maximum_bytes: self.limits.maximum_input_bytes,
            });
        }
        Ok(())
    }

    fn check_cancellation(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeTextRegexError> {
        cancellation
            .check()
            .map_err(|_| NativeTextRegexError::Cancelled)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum NativeTextRegexReplacementPart {
    Literal(String),
    GroupIndex(usize),
    GroupName(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NativeTextRegexReplacement {
    parts: Vec<NativeTextRegexReplacementPart>,
}

impl NativeTextRegexReplacement {
    fn checked(replacement: &str, regex: &Regex) -> Result<Self, NativeTextRegexError> {
        if replacement.len() > NATIVE_TEXT_REGEX_MAX_PATTERN_BYTES {
            return Err(NativeTextRegexError::InvalidReplacement(format!(
                "replacement is {} bytes, above the {}-byte limit",
                replacement.len(),
                NATIVE_TEXT_REGEX_MAX_PATTERN_BYTES
            )));
        }
        let mut parts = Vec::new();
        let mut literal = String::new();
        let characters = replacement.chars().collect::<Vec<_>>();
        let mut position = 0usize;
        while let Some(character) = characters.get(position).copied() {
            position += 1;
            if character != '\\' {
                literal.push(character);
                continue;
            }
            let escaped = characters.get(position).copied().ok_or_else(|| {
                NativeTextRegexError::InvalidReplacement(
                    "replacement ends with an unescaped backslash".to_owned(),
                )
            })?;
            position += 1;
            match escaped {
                '\\' => literal.push('\\'),
                'n' => literal.push('\n'),
                'r' => literal.push('\r'),
                't' => literal.push('\t'),
                'f' => literal.push('\u{000c}'),
                'v' => literal.push('\u{000b}'),
                'a' => literal.push('\u{0007}'),
                'b' => literal.push('\u{0008}'),
                'g' => {
                    flush_literal(&mut parts, &mut literal);
                    if characters.get(position) != Some(&'<') {
                        return Err(NativeTextRegexError::InvalidReplacement(
                            "\\g must be followed by <name> or <number>".to_owned(),
                        ));
                    }
                    position += 1;
                    let mut group = String::new();
                    let mut terminated = false;
                    while let Some(character) = characters.get(position).copied() {
                        position += 1;
                        if character == '>' {
                            terminated = true;
                            break;
                        }
                        group.push(character);
                    }
                    if !terminated || group.is_empty() {
                        return Err(NativeTextRegexError::InvalidReplacement(
                            "\\g group reference is empty or unterminated".to_owned(),
                        ));
                    }
                    if group.chars().all(|character| character.is_ascii_digit()) {
                        let index = group.parse::<usize>().map_err(|_| {
                            NativeTextRegexError::InvalidReplacement(
                                "numeric group reference is too large".to_owned(),
                            )
                        })?;
                        validate_group_index(regex, index)?;
                        parts.push(NativeTextRegexReplacementPart::GroupIndex(index));
                    } else if is_python_group_name(&group) {
                        validate_group_name(regex, &group)?;
                        parts.push(NativeTextRegexReplacementPart::GroupName(group));
                    } else {
                        return Err(NativeTextRegexError::InvalidReplacement(
                            "named group reference is not a valid identifier".to_owned(),
                        ));
                    }
                }
                '0' => {
                    flush_literal(&mut parts, &mut literal);
                    let mut digits = String::from('0');
                    while digits.len() < 3
                        && characters
                            .get(position)
                            .is_some_and(|character| matches!(character, '0'..='7'))
                    {
                        digits.push(characters[position]);
                        position += 1;
                    }
                    let value = u8::from_str_radix(&digits, 8).map_err(|_| {
                        NativeTextRegexError::InvalidReplacement(
                            "octal replacement escape is outside the byte range".to_owned(),
                        )
                    })?;
                    literal.push(char::from(value));
                }
                '1'..='9' => {
                    flush_literal(&mut parts, &mut literal);
                    let three_digit_octal = characters
                        .get(position..position.saturating_add(2))
                        .is_some_and(|suffix| {
                            suffix.len() == 2
                                && matches!(escaped, '1'..='7')
                                && suffix
                                    .iter()
                                    .all(|character| matches!(character, '0'..='7'))
                        });
                    if three_digit_octal {
                        let digits = [escaped, characters[position], characters[position + 1]]
                            .into_iter()
                            .collect::<String>();
                        position += 2;
                        let value = u8::from_str_radix(&digits, 8).map_err(|_| {
                            NativeTextRegexError::InvalidReplacement(
                                "octal replacement escape is outside the byte range".to_owned(),
                            )
                        })?;
                        literal.push(char::from(value));
                    } else {
                        let mut digits = String::from(escaped);
                        if characters.get(position).is_some_and(char::is_ascii_digit) {
                            digits.push(characters[position]);
                            position += 1;
                        }
                        let index = digits.parse::<usize>().map_err(|_| {
                            NativeTextRegexError::InvalidReplacement(
                                "numeric group reference is too large".to_owned(),
                            )
                        })?;
                        validate_group_index(regex, index)?;
                        parts.push(NativeTextRegexReplacementPart::GroupIndex(index));
                    }
                }
                character if character.is_ascii_alphabetic() => {
                    return Err(NativeTextRegexError::InvalidReplacement(format!(
                        "unsupported replacement escape \\{character}"
                    )));
                }
                character => {
                    literal.push('\\');
                    literal.push(character);
                }
            }
        }
        flush_literal(&mut parts, &mut literal);
        Ok(Self { parts })
    }

    fn append_expansion(
        &self,
        captures: &fancy_regex::Captures<'_>,
        output: &mut String,
    ) -> Result<(), NativeTextRegexError> {
        for part in &self.parts {
            let value = match part {
                NativeTextRegexReplacementPart::Literal(value) => Some(value.as_str()),
                NativeTextRegexReplacementPart::GroupIndex(index) => {
                    captures.get(*index).map(|matched| matched.as_str())
                }
                NativeTextRegexReplacementPart::GroupName(name) => {
                    captures.name(name).map(|matched| matched.as_str())
                }
            };
            if let Some(value) = value {
                append_bounded(output, value)?;
            }
        }
        Ok(())
    }
}

fn validate_group_index(regex: &Regex, index: usize) -> Result<(), NativeTextRegexError> {
    if index < regex.captures_len() {
        return Ok(());
    }
    Err(NativeTextRegexError::InvalidReplacement(format!(
        "invalid group reference {index}"
    )))
}

fn validate_group_name(regex: &Regex, name: &str) -> Result<(), NativeTextRegexError> {
    if regex
        .capture_names()
        .flatten()
        .any(|candidate| candidate == name)
    {
        return Ok(());
    }
    Err(NativeTextRegexError::InvalidReplacement(format!(
        "unknown group name `{name}`"
    )))
}

fn flush_literal(parts: &mut Vec<NativeTextRegexReplacementPart>, literal: &mut String) {
    if !literal.is_empty() {
        parts.push(NativeTextRegexReplacementPart::Literal(std::mem::take(
            literal,
        )));
    }
}

fn is_python_group_name(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|character| character == '_' || character.is_alphabetic())
        && characters.all(|character| character == '_' || character.is_alphanumeric())
}

fn append_bounded(output: &mut String, value: &str) -> Result<(), NativeTextRegexError> {
    let result_bytes =
        output
            .len()
            .checked_add(value.len())
            .ok_or(NativeTextRegexError::ResultLimit {
                maximum_bytes: NATIVE_TEXT_REGEX_MAX_RESULT_BYTES,
            })?;
    if result_bytes > NATIVE_TEXT_REGEX_MAX_RESULT_BYTES {
        return Err(NativeTextRegexError::ResultLimit {
            maximum_bytes: NATIVE_TEXT_REGEX_MAX_RESULT_BYTES,
        });
    }
    output.push_str(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn python_style_captures_lookaround_and_backreferences_are_preserved()
    -> Result<(), Box<dyn Error>> {
        let regex = NativeTextRegex::checked(
            r"(?P<word>\w+)\s+(?P=word)(?=!)",
            NativeTextRegexFlags::default(),
        )?;
        let captures = regex.capture_rows("go go! stop", &CancellationToken::default())?;
        assert_eq!(regex.capture_count(), 2);
        assert_eq!(captures.capture_count(), 2);
        assert_eq!(
            captures.rows(),
            &[vec![Some("go go".to_owned()), Some("go".to_owned())]]
        );
        Ok(())
    }

    #[test]
    fn flags_unicode_and_optional_capture_rows_are_exact() -> Result<(), Box<dyn Error>> {
        let regex = NativeTextRegex::checked(
            r"^(?<name>café).(x)?$",
            NativeTextRegexFlags {
                case_insensitive: true,
                multi_line: true,
                dot_matches_new_line: true,
            },
        )?;
        let captures = regex.capture_rows("CAFÉ\n\n", &CancellationToken::default())?;
        assert_eq!(
            captures.rows(),
            &[vec![
                Some("CAFÉ\n".to_owned()),
                Some("CAFÉ".to_owned()),
                None,
            ]]
        );
        Ok(())
    }

    #[test]
    fn invalid_patterns_and_all_resource_limits_fail_closed() -> Result<(), Box<dyn Error>> {
        assert!(matches!(
            NativeTextRegex::checked("(", NativeTextRegexFlags::default()),
            Err(NativeTextRegexError::InvalidPattern(_))
        ));
        assert!(matches!(
            NativeTextRegex::checked(
                &"a".repeat(NATIVE_TEXT_REGEX_MAX_PATTERN_BYTES + 1),
                NativeTextRegexFlags::default(),
            ),
            Err(NativeTextRegexError::PatternTooLarge { .. })
        ));

        let limits = NativeTextRegexLimits {
            maximum_input_bytes: 8,
            maximum_matches: 1,
            maximum_capture_bytes: 2,
            backtrack_limit: 8,
        };
        let input_limited =
            NativeTextRegex::checked_with_limits("a", NativeTextRegexFlags::default(), limits)?;
        assert_eq!(
            input_limited.is_match("123456789", &CancellationToken::default()),
            Err(NativeTextRegexError::InputTooLarge {
                actual_bytes: 9,
                maximum_bytes: 8,
            })
        );
        assert_eq!(
            input_limited.capture_rows("aa", &CancellationToken::default()),
            Err(NativeTextRegexError::MatchLimit { maximum_matches: 1 })
        );

        let capture_limited =
            NativeTextRegex::checked_with_limits("...", NativeTextRegexFlags::default(), limits)?;
        assert_eq!(
            capture_limited.capture_rows("abc", &CancellationToken::default()),
            Err(NativeTextRegexError::CaptureLimit { maximum_bytes: 2 })
        );

        let backtrack_limited = NativeTextRegex::checked_with_limits(
            r"(x+x+)+(?>y)",
            NativeTextRegexFlags::default(),
            NativeTextRegexLimits {
                maximum_input_bytes: 64,
                backtrack_limit: 1,
                ..limits
            },
        )?;
        assert!(matches!(
            backtrack_limited.is_match("xxxxxxxxxxy", &CancellationToken::default()),
            Err(NativeTextRegexError::ExecutionLimit(_))
        ));
        Ok(())
    }

    #[test]
    fn cancellation_is_checked_before_matching_and_between_capture_rows()
    -> Result<(), Box<dyn Error>> {
        let regex = NativeTextRegex::checked("a", NativeTextRegexFlags::default())?;
        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert_eq!(
            regex.is_match("a", &cancellation),
            Err(NativeTextRegexError::Cancelled)
        );
        assert_eq!(
            regex.capture_rows("aaa", &cancellation),
            Err(NativeTextRegexError::Cancelled)
        );
        Ok(())
    }

    #[test]
    fn python_replacement_templates_preserve_groups_escapes_counts_and_zero_width_matches()
    -> Result<(), Box<dyn Error>> {
        let regex =
            NativeTextRegex::checked(r"(?P<word>\w+)(?=!)|^", NativeTextRegexFlags::default())?;
        assert_eq!(
            regex.replace(
                "go! stop!",
                r"<\g<word>>\n",
                0,
                &CancellationToken::default(),
            )?,
            "<go>\n! <stop>\n!"
        );

        let zero_width = NativeTextRegex::checked(r"(?=a)", NativeTextRegexFlags::default())?;
        assert_eq!(
            zero_width.replace("aba", "X", 0, &CancellationToken::default())?,
            "XabXa"
        );

        let numbered =
            NativeTextRegex::checked(r"([a-z]+)-(\d+)", NativeTextRegexFlags::default())?;
        assert_eq!(
            numbered.replace("a-1 b-2", r"\2:\1:\\", 1, &CancellationToken::default(),)?,
            "1:a:\\ b-2"
        );
        Ok(())
    }

    #[test]
    fn python_replacement_octal_and_invalid_group_references_are_exact()
    -> Result<(), Box<dyn Error>> {
        let regex = NativeTextRegex::checked(r"(?P<word>a)(b)?", NativeTextRegexFlags::default())?;
        assert_eq!(
            regex.replace(
                "a",
                "\\0-\\123-\\111-\\08-\\g<0>-\\g<word>-\\2",
                0,
                &CancellationToken::default(),
            )?,
            "\0-S-I-\08-a-a-"
        );
        assert!(matches!(
            regex.replace("a", r"\3", 0, &CancellationToken::default()),
            Err(NativeTextRegexError::InvalidReplacement(message))
                if message.contains("invalid group reference 3")
        ));
        assert!(matches!(
            regex.replace("a", r"\g<missing>", 0, &CancellationToken::default()),
            Err(NativeTextRegexError::InvalidReplacement(message))
                if message.contains("unknown group name")
        ));
        assert!(matches!(
            regex.replace("a", r"\400", 0, &CancellationToken::default()),
            Err(NativeTextRegexError::InvalidReplacement(_))
        ));
        Ok(())
    }

    #[test]
    fn replacement_result_limit_and_concurrent_cancellation_fail_closed()
    -> Result<(), Box<dyn Error>> {
        let regex = NativeTextRegex::checked("a", NativeTextRegexFlags::default())?;
        assert_eq!(
            regex.replace(
                &"a".repeat(1_100),
                &"x".repeat(NATIVE_TEXT_REGEX_MAX_PATTERN_BYTES),
                0,
                &CancellationToken::default(),
            ),
            Err(NativeTextRegexError::ResultLimit {
                maximum_bytes: NATIVE_TEXT_REGEX_MAX_RESULT_BYTES,
            })
        );

        let cancellation = CancellationToken::default();
        let worker_cancellation = cancellation.clone();
        let worker = std::thread::spawn(move || {
            regex.replace(
                &"a".repeat(NATIVE_TEXT_REGEX_MAX_INPUT_BYTES),
                "b",
                0,
                &worker_cancellation,
            )
        });
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(cancellation.cancel());
        assert_eq!(
            worker.join().map_err(|_| "replacement worker panicked")?,
            Err(NativeTextRegexError::Cancelled)
        );
        Ok(())
    }

    #[test]
    fn invalid_replacement_limits_and_cancellation_fail_closed() -> Result<(), Box<dyn Error>> {
        let regex = NativeTextRegex::checked("a", NativeTextRegexFlags::default())?;
        assert!(matches!(
            regex.replace("a", r"\g<", 0, &CancellationToken::default()),
            Err(NativeTextRegexError::InvalidReplacement(_))
        ));
        assert!(matches!(
            regex.replace("a", r"\q", 0, &CancellationToken::default()),
            Err(NativeTextRegexError::InvalidReplacement(_))
        ));

        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        assert_eq!(
            regex.replace("a", "b", 0, &cancellation),
            Err(NativeTextRegexError::Cancelled)
        );
        Ok(())
    }
}
