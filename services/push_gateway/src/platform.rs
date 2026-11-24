pub mod apns;
pub mod app_attest;
pub mod grant;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ApprovedPushProfile {
    BuzzIosProduction,
    BuzzIosSandbox,
}

impl ApprovedPushProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BuzzIosProduction => "buzz-ios-production",
            Self::BuzzIosSandbox => "buzz-ios-sandbox",
        }
    }

    pub fn parse(value: &str) -> Result<Self, UnsupportedPushProvider> {
        match value {
            "buzz-ios-production" => Ok(Self::BuzzIosProduction),
            "buzz-ios-sandbox" => Ok(Self::BuzzIosSandbox),
            _ => Err(UnsupportedPushProvider),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("push provider profile is not approved")]
pub struct UnsupportedPushProvider;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_adr_005_apns_profiles_are_advertised() {
        assert_eq!(
            ApprovedPushProfile::parse("buzz-ios-production"),
            Ok(ApprovedPushProfile::BuzzIosProduction)
        );
        assert_eq!(
            ApprovedPushProfile::parse("buzz-ios-sandbox"),
            Ok(ApprovedPushProfile::BuzzIosSandbox)
        );
        for unsupported in [
            "fcm",
            "unified-push",
            "webhook",
            "desktop",
            "zed-ios-production",
        ] {
            assert_eq!(
                ApprovedPushProfile::parse(unsupported),
                Err(UnsupportedPushProvider)
            );
        }
    }
}
