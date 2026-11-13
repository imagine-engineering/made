mod plan_usage;

pub use plan_usage::State;

use language_model::{LanguageModelProviderId, LanguageModelProviderName};

pub const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("openai-subscribed");
pub const PROVIDER_NAME: LanguageModelProviderName =
    LanguageModelProviderName::new("ChatGPT Subscription");
