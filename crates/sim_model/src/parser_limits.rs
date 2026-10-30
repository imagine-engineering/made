use serde::{Deserialize, Serialize};

pub const PARSER_LIMITS_VERSION: u32 = 1;
pub const PARSER_DECODED_ALLOCATION_MULTIPLIER: u64 = 8;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ParserLimits {
    pub version: u32,
    pub manifest_bytes: u64,
    pub maximum_depth: u32,
    pub maximum_tensors: u64,
    pub maximum_tensor_bytes: u64,
    pub maximum_aggregate_tensor_bytes: u64,
    pub maximum_name_bytes: u64,
    pub maximum_archive_entries: u64,
    pub maximum_metadata_values: u64,
}

impl Default for ParserLimits {
    fn default() -> Self {
        Self {
            version: PARSER_LIMITS_VERSION,
            manifest_bytes: 16 * 1024 * 1024,
            maximum_depth: 64,
            maximum_tensors: 1_000_000,
            maximum_tensor_bytes: 64 * 1024 * 1024 * 1024,
            maximum_aggregate_tensor_bytes: 1024 * 1024 * 1024 * 1024,
            maximum_name_bytes: 64 * 1024,
            maximum_archive_entries: 1_001_024,
            maximum_metadata_values: 1_000_000,
        }
    }
}

impl ParserLimits {
    pub fn maximum_decoded_allocation_bytes(&self) -> Result<u64, ParserLimitError> {
        self.manifest_bytes
            .checked_mul(PARSER_DECODED_ALLOCATION_MULTIPLIER)
            .ok_or(ParserLimitError::DecodedAllocationOverflow)
    }

    pub fn validate(&self) -> Result<(), ParserLimitError> {
        if self.version != PARSER_LIMITS_VERSION {
            return Err(ParserLimitError::UnsupportedVersion(self.version));
        }

        for (name, value) in [
            ("manifest_bytes", self.manifest_bytes),
            ("maximum_depth", u64::from(self.maximum_depth)),
            ("maximum_tensors", self.maximum_tensors),
            ("maximum_tensor_bytes", self.maximum_tensor_bytes),
            (
                "maximum_aggregate_tensor_bytes",
                self.maximum_aggregate_tensor_bytes,
            ),
            ("maximum_name_bytes", self.maximum_name_bytes),
            ("maximum_archive_entries", self.maximum_archive_entries),
            ("maximum_metadata_values", self.maximum_metadata_values),
        ] {
            if value == 0 {
                return Err(ParserLimitError::Zero(name));
            }
        }

        if self.maximum_tensor_bytes > self.maximum_aggregate_tensor_bytes {
            return Err(ParserLimitError::TensorExceedsAggregate);
        }
        if self.maximum_tensors > self.maximum_archive_entries {
            return Err(ParserLimitError::TensorCountExceedsArchiveEntries);
        }
        self.maximum_decoded_allocation_bytes()?;
        Ok(())
    }

    pub(crate) fn check(
        &self,
        kind: &'static str,
        actual: u64,
        maximum: u64,
    ) -> Result<(), ParserLimitError> {
        if actual > maximum {
            Err(ParserLimitError::Exceeded {
                kind,
                actual,
                maximum,
            })
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ParserLimitError {
    #[error("unsupported parser-limit fixture version {0}")]
    UnsupportedVersion(u32),
    #[error("parser limit {0} must be non-zero")]
    Zero(&'static str),
    #[error("maximum tensor bytes exceeds the aggregate tensor-byte limit")]
    TensorExceedsAggregate,
    #[error("maximum tensor count exceeds the archive-entry limit")]
    TensorCountExceedsArchiveEntries,
    #[error("decoded-allocation limit overflowed")]
    DecodedAllocationOverflow,
    #[error("{kind} limit exceeded: {actual} > {maximum}")]
    Exceeded {
        kind: &'static str,
        actual: u64,
        maximum: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normative_defaults_are_versioned_and_valid() {
        let limits = ParserLimits::default();
        assert_eq!(limits.version, PARSER_LIMITS_VERSION);
        assert_eq!(limits.manifest_bytes, 16 * 1024 * 1024);
        assert_eq!(limits.maximum_depth, 64);
        assert_eq!(limits.maximum_tensors, 1_000_000);
        assert_eq!(limits.maximum_tensor_bytes, 64 * 1024 * 1024 * 1024);
        assert_eq!(limits.maximum_name_bytes, 64 * 1024);
        assert_eq!(
            limits.maximum_decoded_allocation_bytes(),
            Ok(128 * 1024 * 1024)
        );
        assert_eq!(limits.validate(), Ok(()));
    }

    #[test]
    fn inconsistent_limits_are_rejected() {
        let limits = ParserLimits {
            maximum_aggregate_tensor_bytes: 1,
            ..ParserLimits::default()
        };
        assert_eq!(
            limits.validate(),
            Err(ParserLimitError::TensorExceedsAggregate)
        );
    }
}
