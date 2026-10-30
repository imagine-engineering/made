pub mod agent_provider_discovery;
pub mod agent_provider_lifecycle;
pub mod agent_provider_protocol;
pub mod json_log;
#[cfg(feature = "multiplayer-tools")]
pub mod mesh;
pub mod protocol;
pub mod proxy;
pub mod remote_client;
pub mod remote_identity;
mod transport;

#[cfg(target_os = "windows")]
pub use remote_client::OpenWslPath;
pub use remote_client::{
    CommandTemplate, ConnectionIdentifier, ConnectionState, Interactive, ProxyMode, RemoteArch,
    RemoteClient, RemoteClientDelegate, RemoteClientEvent, RemoteConnection,
    RemoteConnectionOptions, RemoteOs, RemotePlatform, RemoteRequestActivity, connect,
    has_active_connection,
};
pub use remote_identity::{
    RemoteConnectionIdentity, remote_connection_identity, same_remote_connection_identity,
};
pub use transport::docker::{
    DockerConnectionOptions, DockerRecipeTestOptions, connect_recipe_test,
};
pub use transport::ssh::{SshConnectionOptions, SshPortForwardOption};
pub use transport::wsl::WslConnectionOptions;
#[cfg(target_os = "windows")]
pub use transport::wsl::wsl_path_to_windows_path;

#[cfg(all(any(test, feature = "test-support"), target_os = "macos"))]
pub use transport::mock::ProcessRemoteConnection;
#[cfg(any(test, feature = "test-support"))]
pub use transport::mock::{
    MockConnection, MockConnectionOptions, MockConnectionRegistry, MockDelegate,
};
