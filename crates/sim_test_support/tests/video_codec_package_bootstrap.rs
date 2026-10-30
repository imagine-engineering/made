use anyhow::{Context as _, Result, anyhow};
use sim_runtime::{
    NativeGeneralVideoCodecPackageSettings, NativeVideoCodecWorkerServices,
    certify_general_video_codec_package,
};
use sim_tensor::{BackendWorkspaceAuthority, CancellationToken};
use std::{fs, path::Path, sync::Arc};

const FIXTURE_SIGNER: &str = "sim.fixture.general-video";
const FIXTURE_PUBLIC_KEY_HEX: &str =
    "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";

fn fixture_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/video/codec-package-bootstrap")
}

fn fixture_settings(root: &Path) -> Result<NativeGeneralVideoCodecPackageSettings> {
    NativeGeneralVideoCodecPackageSettings::from_fixture_authority(
        root,
        FIXTURE_SIGNER,
        FIXTURE_PUBLIC_KEY_HEX,
    )
    .map_err(|error| anyhow!(error))
}

#[test]
fn signed_general_video_package_certifies_and_relocates_stably() -> Result<()> {
    for relative_path in [
        "lib/libavcodec.so.61",
        "lib/libavfilter.so.10",
        "lib/libavformat.so.61",
        "lib/libavutil.so.59",
        "lib/libswresample.so.5",
        "lib/libswscale.so.8",
        "lib/libSvtAv1Enc.so.2",
        "lib/libvpx.so.9",
        "lib/libx264.so.164",
    ] {
        let bytes = fs::read(fixture_root().join(relative_path))?;
        let version = bytes
            .get(20..24)
            .and_then(|bytes| <[u8; 4]>::try_from(bytes).ok())
            .map(u32::from_le_bytes)
            .ok_or_else(|| anyhow!("fixture ELF header is truncated: {relative_path}"))?;
        let header_size = bytes
            .get(52..54)
            .and_then(|bytes| <[u8; 2]>::try_from(bytes).ok())
            .map(u16::from_le_bytes)
            .ok_or_else(|| anyhow!("fixture ELF header is truncated: {relative_path}"))?;
        assert_eq!(version, 1, "fixture ELF version drifted: {relative_path}");
        assert_eq!(header_size, 64, "fixture ELF size drifted: {relative_path}");
    }

    let cancellation = CancellationToken::default();
    let first =
        certify_general_video_codec_package(&fixture_settings(&fixture_root())?, &cancellation)?;
    assert_eq!(first.libraries().len(), 6);
    assert_eq!(first.primary_certificate_count(), 6);
    assert_eq!(first.dependency_certificate_count(), 3);
    assert!(first.retained_image_bytes() > 0);
    assert!(first.startup_resident_bytes() > first.retained_image_bytes());
    assert_eq!(first.codec_scratch_bytes(), 2 * 1024 * 1024 * 1024);

    let relocated = tempfile::tempdir()?;
    copy_tree(&fixture_root(), relocated.path())?;
    let second =
        certify_general_video_codec_package(&fixture_settings(relocated.path())?, &cancellation)?;
    assert_eq!(first.semantic_identity(), second.semantic_identity());
    assert_eq!(first.libraries(), second.libraries());
    assert_eq!(
        first.startup_resident_bytes(),
        second.startup_resident_bytes()
    );
    Ok(())
}

#[test]
fn signed_general_video_package_rejects_tree_and_payload_mutations() -> Result<()> {
    let cancellation = CancellationToken::default();

    let tampered = tempfile::tempdir()?;
    copy_tree(&fixture_root(), tampered.path())?;
    fs::write(
        tampered.path().join("licenses/ffmpeg-license.txt"),
        b"tampered\n",
    )?;
    assert!(
        certify_general_video_codec_package(&fixture_settings(tampered.path())?, &cancellation,)
            .is_err()
    );

    let extra = tempfile::tempdir()?;
    copy_tree(&fixture_root(), extra.path())?;
    fs::write(extra.path().join("unexpected.bin"), b"extra")?;
    assert!(
        certify_general_video_codec_package(&fixture_settings(extra.path())?, &cancellation)
            .is_err()
    );

    let missing = tempfile::tempdir()?;
    copy_tree(&fixture_root(), missing.path())?;
    fs::remove_file(missing.path().join("lib/libavfilter.so.10"))?;
    assert!(
        certify_general_video_codec_package(&fixture_settings(missing.path())?, &cancellation)
            .is_err()
    );

    for relative_path in [
        "package-coverage.sha256",
        "package-signature.json",
        "dependency-contract-v1.signature.json",
    ] {
        let mutated = tempfile::tempdir()?;
        copy_tree(&fixture_root(), mutated.path())?;
        let path = mutated.path().join(relative_path);
        let mut bytes = fs::read(&path)?;
        let first = bytes
            .first_mut()
            .ok_or_else(|| anyhow!("fixture file is unexpectedly empty: {relative_path}"))?;
        *first = if *first == b'0' { b'1' } else { b'0' };
        fs::write(&path, bytes)?;
        assert!(
            certify_general_video_codec_package(&fixture_settings(mutated.path())?, &cancellation,)
                .is_err(),
            "mutation of {relative_path} must be rejected"
        );
    }

    let wrong_signer = NativeGeneralVideoCodecPackageSettings::from_fixture_authority(
        &fixture_root(),
        "sim.fixture.general-video.changed",
        FIXTURE_PUBLIC_KEY_HEX,
    )
    .map_err(|error| anyhow!(error))?;
    assert!(certify_general_video_codec_package(&wrong_signer, &cancellation).is_err());
    let wrong_key = NativeGeneralVideoCodecPackageSettings::from_fixture_authority(
        &fixture_root(),
        FIXTURE_SIGNER,
        &"11".repeat(32),
    )
    .map_err(|error| anyhow!(error))?;
    assert!(certify_general_video_codec_package(&wrong_key, &cancellation).is_err());

    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;

        let linked = tempfile::tempdir()?;
        copy_tree(&fixture_root(), linked.path())?;
        let license_path = linked.path().join("licenses/ffmpeg-license.txt");
        fs::remove_file(&license_path)?;
        symlink(
            fixture_root().join("licenses/ffmpeg-license.txt"),
            &license_path,
        )?;
        let error = match certify_general_video_codec_package(
            &fixture_settings(linked.path())?,
            &cancellation,
        ) {
            Ok(_) => return Err(anyhow!("a symlinked package file was accepted")),
            Err(error) => error,
        };
        assert!(
            !error
                .to_string()
                .contains(&linked.path().display().to_string())
        );
    }

    let cancellation = CancellationToken::default();
    cancellation.cancel();
    assert!(matches!(
        certify_general_video_codec_package(&fixture_settings(&fixture_root())?, &cancellation,),
        Err(sim_runtime::GeneralVideoCodecPackageError::Cancelled)
    ));
    Ok(())
}

#[test]
fn general_video_worker_publication_uses_one_ready_actor_and_three_shared_ports() {
    let service_source = include_str!("../../sim_runtime/src/native_video_codec_service.rs");
    let package_source = include_str!("../../sim_runtime/src/native_video_codec_package.rs");
    let worker_source = include_str!("../../sim_worker/src/sim_worker.rs");
    let worker_production = worker_source
        .split("#[cfg(test)]")
        .next()
        .expect("worker production source must precede tests");

    let actor = service_source
        .find("let actor = NativeLtxvCodecThreadService::start_general")
        .expect("the bundle must consume the closure into one actor");
    let h264 = service_source
        .find("let component_h264_mp4_backing_service")
        .expect("the H.264 port must be constructed");
    let cache = service_source
        .find("let cache_configuration_sha256 = worker_service_cache_configuration_identity")
        .expect("the cache identity must aggregate all ports");
    let publish = service_source[cache..]
        .find("Ok(Self {")
        .map(|offset| cache + offset)
        .expect("the bundle must publish atomically");
    assert!(actor < h264 && h264 < cache && cache < publish);
    assert!(package_source.contains("manifest.service_limits.actor_capacity != 1"));

    let bundle = worker_source
        .find("let video_codec_worker_services =")
        .expect("worker must retain the bundle");
    let worker_loop = worker_source
        .find("'worker: loop")
        .expect("worker loop must be explicit");
    let attach = worker_source
        .find("with_ltxv_preprocess_service")
        .expect("all executor branches must attach the shared ports");
    assert!(bundle < worker_loop && worker_loop < attach);
    for port in [
        "with_ltxv_preprocess_service",
        "with_webm_encode_service",
        "with_component_h264_mp4_backing_service",
    ] {
        assert_eq!(worker_production.matches(port).count(), 1, "{port}");
    }
}

#[test]
fn general_video_worker_bundle_is_ready_or_typed_unsupported() -> Result<()> {
    let cancellation = CancellationToken::default();
    let closure =
        certify_general_video_codec_package(&fixture_settings(&fixture_root())?, &cancellation)?;
    let (backend, _authority) = BackendWorkspaceAuthority::create_backend(4 * 1024 * 1024 * 1024)?;
    let result = NativeVideoCodecWorkerServices::start(closure, Arc::new(backend), &cancellation);
    #[cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))]
    {
        let services = result?;
        assert_eq!(services.codec_scratch_bytes(), 2 * 1024 * 1024 * 1024);
        assert_eq!(services.cache_configuration_sha256().len(), 64);
        services.shutdown()?;
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu")))]
    assert!(matches!(
        result,
        Err(sim_runtime::NativeVideoCodecWorkerServicesError::UnsupportedTarget)
    ));
    Ok(())
}

fn copy_tree(source: &Path, destination: &Path) -> Result<()> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            fs::create_dir_all(&destination_path)?;
            copy_tree(&source_path, &destination_path)?;
        } else if entry.file_type()?.is_file() {
            fs::copy(&source_path, &destination_path)
                .with_context(|| format!("copy fixture {}", source_path.display()))?;
        } else {
            return Err(anyhow!("fixture contains a non-regular entry"));
        }
    }
    Ok(())
}
