use sim_types::CancellationToken;
use thiserror::Error;

use crate::{
    NativeUpscaleCanonicalStateKeys, NativeUpscaleContractError, NativeUpscaleDetection,
    NativeUpscaleStateDictionaryLayout, compiled_native_upscale_contract,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeUpscaleUnavailableReason {
    MissingIndividualLicense,
    ReferenceOnlyExtraArchitecture,
}

impl NativeUpscaleUnavailableReason {
    pub const fn diagnostic(self) -> &'static str {
        match self {
            Self::MissingIndividualLicense => "rejected-missing-individual-license-artifact",
            Self::ReferenceOnlyExtraArchitecture => "rejected-reference-only-extra-architecture",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeUpscaleUnavailable {
    architecture_id: String,
    ordinal: usize,
    reason: NativeUpscaleUnavailableReason,
}

impl NativeUpscaleUnavailable {
    pub fn architecture_id(&self) -> &str {
        &self.architecture_id
    }

    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub const fn reason(&self) -> NativeUpscaleUnavailableReason {
        self.reason
    }

    pub const fn diagnostic(&self) -> &'static str {
        self.reason.diagnostic()
    }
}

#[derive(Debug, Error)]
pub enum NativeUpscaleModelError {
    #[error(transparent)]
    Contract(#[from] NativeUpscaleContractError),
    #[error(
        "native upscale architecture {architecture_id} is unavailable: {}",
        reason.diagnostic()
    )]
    Unavailable {
        architecture_id: String,
        ordinal: usize,
        reason: NativeUpscaleUnavailableReason,
    },
}

impl NativeUpscaleModelError {
    pub fn unavailable(&self) -> Option<NativeUpscaleUnavailable> {
        match self {
            Self::Unavailable {
                architecture_id,
                ordinal,
                reason,
            } => Some(NativeUpscaleUnavailable {
                architecture_id: architecture_id.clone(),
                ordinal: *ordinal,
                reason: *reason,
            }),
            Self::Contract(_) => None,
        }
    }
}

pub enum NativeUpscaleModelResource {}

impl NativeUpscaleModelResource {
    pub fn checked<I, S>(
        layout: NativeUpscaleStateDictionaryLayout,
        state_keys: I,
        cancellation: &CancellationToken,
    ) -> Result<Self, NativeUpscaleModelError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        cancellation
            .check()
            .map_err(|_| NativeUpscaleContractError::Cancelled)?;
        let state_keys =
            NativeUpscaleCanonicalStateKeys::checked(layout, state_keys, cancellation)?;
        let contract = compiled_native_upscale_contract()?;
        match contract.detect_canonical_state_keys(&state_keys, cancellation)? {
            NativeUpscaleDetection::Unavailable { architecture } => {
                Err(unavailable_error(architecture))
            }
        }
    }
}

fn unavailable_error(
    architecture: &crate::NativeUpscaleArchitectureContract,
) -> NativeUpscaleModelError {
    let reason = if architecture.origin == "main" {
        NativeUpscaleUnavailableReason::MissingIndividualLicense
    } else {
        NativeUpscaleUnavailableReason::ReferenceOnlyExtraArchitecture
    };
    NativeUpscaleModelError::Unavailable {
        architecture_id: architecture.architecture_id.clone(),
        ordinal: architecture.ordinal,
        reason,
    }
}
