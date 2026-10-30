#[path = "support/provider_worker_stream_bridge.rs"]
mod provider_worker_stream_bridge;

#[test]
fn first_valid_grant_reaches_verified_materialization_and_finalization_ack()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(provider_worker_stream_bridge::task425_entrypoint_count(), 3);
    provider_worker_stream_bridge::first_valid_grant_reaches_verified_materialization_and_finalization_ack()
}

#[test]
fn cancellation_between_response_chunks_revokes_without_publication_and_clean_retry_is_unique()
-> Result<(), Box<dyn std::error::Error>> {
    provider_worker_stream_bridge::cancellation_between_response_chunks_revokes_without_publication_and_clean_retry_is_unique()
}

#[test]
fn worker_crash_revokes_pending_route_and_restarts_without_duplicate_publication()
-> Result<(), Box<dyn std::error::Error>> {
    provider_worker_stream_bridge::worker_crash_revokes_pending_route_and_restarts_without_duplicate_publication()
}
