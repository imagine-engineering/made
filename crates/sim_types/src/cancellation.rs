use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use thiserror::Error;

#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn cancel(&self) -> bool {
        !self.cancelled.swap(true, Ordering::AcqRel)
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub fn check(&self) -> Result<(), CancellationError> {
        if self.is_cancelled() {
            Err(CancellationError)
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[error("operation was cancelled")]
pub struct CancellationError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn val_cancel_001_clones_share_one_monotonic_state() {
        let token = CancellationToken::default();
        let clone = token.clone();

        assert_eq!(token.check(), Ok(()));
        assert!(clone.cancel());
        assert!(!token.cancel());
        assert_eq!(token.check(), Err(CancellationError));
        assert!(clone.is_cancelled());
    }

    #[test]
    fn val_cancel_001_independent_tokens_do_not_share_state() {
        let first = CancellationToken::default();
        let second = CancellationToken::default();

        assert!(first.cancel());
        assert!(first.is_cancelled());
        assert!(!second.is_cancelled());
    }
}
