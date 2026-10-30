use std::sync::OnceLock;

use rustls::ClientConfig;
use rustls_platform_verifier::ConfigVerifierExt;

static TLS_CONFIG: OnceLock<rustls::ClientConfig> = OnceLock::new();

pub fn tls_config() -> ClientConfig {
    TLS_CONFIG
        .get_or_init(|| {
            // rustls uses the `aws_lc_rs` provider by default
            // This only errors if the default provider has already
            // been installed. We can ignore this `Result`.
            rustls::crypto::aws_lc_rs::default_provider()
                .install_default()
                .ok();

            ClientConfig::with_platform_verifier()
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::tls_config;
    use rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer, ServerName};
    use rustls::{ClientConfig, ClientConnection, Connection, RootCertStore, ServerConfig};
    use std::io::{Cursor, Error, Read as _, Write as _};
    use std::sync::Arc;

    type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

    fn server_connection() -> TestResult<Connection> {
        let configuration =
            ServerConfig::builder_with_provider(tls_config().crypto_provider().clone())
                .with_protocol_versions(&[&rustls::version::TLS13, &rustls::version::TLS12])?
                .with_no_client_auth()
                .with_single_cert(
                    vec![CertificateDer::from(
                        include_bytes!("../test_data/server.der").to_vec(),
                    )],
                    PrivatePkcs8KeyDer::from(
                        include_bytes!("../test_data/server-key.der").to_vec(),
                    )
                    .into(),
                )?;
        Ok(Connection::Server(rustls::ServerConnection::new(
            Arc::new(configuration),
        )?))
    }

    fn transfer(sender: &mut Connection, receiver: &mut Connection) -> TestResult {
        let mut records = Vec::new();
        sender.write_tls(&mut records)?;
        if !records.is_empty() {
            let read = receiver.read_tls(&mut Cursor::new(&records))?;
            assert_eq!(read, records.len());
            receiver.process_new_packets()?;
        }
        Ok(())
    }

    #[test]
    fn valid_handshakes_require_explicit_fixture_trust() -> TestResult {
        for version in [&rustls::version::TLS13, &rustls::version::TLS12] {
            let mut roots = RootCertStore::empty();
            roots.add(CertificateDer::from(
                include_bytes!("../test_data/server.der").to_vec(),
            ))?;
            let configuration =
                ClientConfig::builder_with_provider(tls_config().crypto_provider().clone())
                    .with_protocol_versions(&[version])?
                    .with_root_certificates(roots)
                    .with_no_client_auth();
            let mut client = Connection::Client(ClientConnection::new(
                Arc::new(configuration),
                ServerName::try_from("localhost")?,
            )?);
            let mut server = server_connection()?;
            for _ in 0..8 {
                transfer(&mut client, &mut server)?;
                transfer(&mut server, &mut client)?;
                if !client.is_handshaking() && !server.is_handshaking() {
                    break;
                }
            }
            assert!(!client.is_handshaking());
            assert!(!server.is_handshaking());
            assert_eq!(client.protocol_version(), Some(version.version));
            let message = b"Made TLS regression";
            client.writer().write_all(message)?;
            transfer(&mut client, &mut server)?;
            let mut received = vec![0; message.len()];
            server.reader().read_exact(&mut received)?;
            assert_eq!(received, message);

            let mut untrusted_client = Connection::Client(ClientConnection::new(
                Arc::new(tls_config()),
                ServerName::try_from("localhost")?,
            )?);
            let mut server = server_connection()?;
            transfer(&mut untrusted_client, &mut server)?;
            let error = transfer(&mut server, &mut untrusted_client)
                .expect_err("production verifier trusted the public self-signed fixture");
            assert!(
                matches!(
                    error.downcast_ref::<rustls::Error>(),
                    Some(rustls::Error::InvalidCertificate(_))
                ),
                "failed for a reason other than certificate verification: {error}"
            );
            assert!(untrusted_client.is_handshaking());
        }
        Ok(())
    }

    #[test]
    fn rejects_plaintext_handshake_after_tls13_server_hello() -> TestResult {
        let mut client = Connection::Client(ClientConnection::new(
            Arc::new(tls_config()),
            ServerName::try_from("localhost")?,
        )?);
        let mut server = server_connection()?;
        transfer(&mut client, &mut server)?;
        let mut flight = Vec::new();
        server.write_tls(&mut flight)?;
        let [
            content_type,
            major,
            minor,
            length_high,
            length_low,
            body @ ..,
        ] = flight.as_slice()
        else {
            return Err(Error::other("server flight has no TLS record header").into());
        };
        assert_eq!(*content_type, 22);
        let length = usize::from(u16::from_be_bytes([*length_high, *length_low]));
        let server_hello = body
            .get(..length)
            .ok_or_else(|| Error::other("server hello record is truncated"))?;

        // EncryptedExtensions must follow the key change in an encrypted record, even if
        // a complete extra handshake message fits after ServerHello in its plaintext record.
        let encrypted_extensions = [8, 0, 0, 2, 0, 0];
        let altered_length = u16::try_from(length + encrypted_extensions.len())?;
        let mut altered = vec![*content_type, *major, *minor];
        altered.extend_from_slice(&altered_length.to_be_bytes());
        altered.extend_from_slice(server_hello);
        altered.extend_from_slice(&encrypted_extensions);
        let read = client.read_tls(&mut Cursor::new(&altered))?;
        assert_eq!(read, altered.len());
        let result = client.process_new_packets();
        assert!(
            matches!(
                &result,
                Err(rustls::Error::PeerMisbehaved(
                    rustls::PeerMisbehaved::KeyEpochWithPendingFragment
                ))
            ),
            "accepted a handshake message across a TLS key change: {result:?}"
        );
        Ok(())
    }
}
