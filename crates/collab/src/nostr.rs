pub const MAX_NOSTR_FRAME_BYTES: usize = 512 * 1024;

#[path = "nostr/auth.rs"]
pub mod auth;
#[path = "nostr/event_ingest.rs"]
pub mod event_ingest;
#[path = "nostr/http.rs"]
pub mod http;
#[path = "nostr/ingress.rs"]
pub mod ingress;
#[path = "nostr/replay.rs"]
pub mod replay;
#[path = "nostr/subscriptions.rs"]
pub mod subscriptions;
