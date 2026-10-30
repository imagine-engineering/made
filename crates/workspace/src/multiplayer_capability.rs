use serde::{Deserialize, Serialize};
use std::{error::Error, fmt};

pub const MULTIPLAYER_TOOLS_CAPABILITY: &str = "multiplayer-tools";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MultiplayerCapabilityAdvertisement {
    pub multiplayer_tools: bool,
}

impl MultiplayerCapabilityAdvertisement {
    pub const fn current_build() -> Self {
        Self {
            multiplayer_tools: multiplayer_tools_available(),
        }
    }
}

pub const fn multiplayer_tools_available() -> bool {
    cfg!(feature = "multiplayer-tools")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MultiplayerCapabilityError {
    NotIncludedInBuild,
}

impl fmt::Display for MultiplayerCapabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotIncludedInBuild => {
                formatter.write_str("multiplayer tools are not included in this build")
            }
        }
    }
}

impl Error for MultiplayerCapabilityError {}

pub fn admit_multiplayer_operation<T>(
    resolve_after_admission: impl FnOnce() -> T,
) -> Result<T, MultiplayerCapabilityError> {
    if !multiplayer_tools_available() {
        return Err(MultiplayerCapabilityError::NotIncludedInBuild);
    }

    Ok(resolve_after_admission())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn advertisement_explicitly_reports_build_capability() {
        let advertisement = MultiplayerCapabilityAdvertisement::current_build();
        assert_eq!(
            advertisement.multiplayer_tools,
            cfg!(feature = "multiplayer-tools")
        );
        assert_eq!(
            serde_json::to_value(advertisement).expect("advertisement should serialize"),
            serde_json::json!({
                "multiplayer_tools": cfg!(feature = "multiplayer-tools"),
            })
        );
    }

    #[cfg(not(feature = "multiplayer-tools"))]
    #[test]
    fn unavailable_operations_reject_before_any_resource_lookup() {
        for target_class in ["missing", "foreign", "denied"] {
            let lookup_started = Cell::new(false);
            let result = admit_multiplayer_operation(|| {
                lookup_started.set(true);
                target_class
            });

            assert_eq!(result, Err(MultiplayerCapabilityError::NotIncludedInBuild));
            assert!(!lookup_started.get());
            assert_eq!(
                result
                    .expect_err("standard builds reject multiplayer operations")
                    .to_string(),
                "multiplayer tools are not included in this build"
            );
        }
    }

    #[cfg(feature = "multiplayer-tools")]
    #[test]
    fn available_operations_resolve_only_after_admission() {
        let lookup_started = Cell::new(false);
        let result = admit_multiplayer_operation(|| {
            lookup_started.set(true);
            "resource"
        });

        assert_eq!(result, Ok("resource"));
        assert!(lookup_started.get());
    }
}
