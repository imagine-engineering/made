use anyhow::{Context as _, Result, anyhow, bail};
use ring::signature::{Ed25519KeyPair, KeyPair as _};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

const FIXTURE_SIGNER: &str = "sim.fixture.general-video";
const FIXTURE_SEED_HEX: &str = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60";
const PACKAGE_DOMAIN: &[u8] = b"zed-sim-general-video-codec-package-v1\0";
const DEPENDENCY_DOMAIN: &[u8] = b"zed-sim-general-video-codec-dependency-contract-v1\0";
const ARCHIVE_SHA256: &str = "40973d44970dbc83ef302b0609f2e74982be2d85916dd2ee7472d30678a7abe6";
const SIGNATURE_SHA256: &str = "9bd1689dce76b109034dcc4765a406e84e8799a2fd857b000c0a4d9744b70617";
const SIGNING_FINGERPRINT: &str = "FCF986EA15E6E293A5644F10B4322F04D67658D8";
const GENERAL_ABI_SHA256: &str = "772b01d7db041e0da57dd7d5a09e1b5ae267804b934b2b533ce1fc62e521089d";
const CODEC_SCRATCH_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Deserialize)]
struct FixtureGeneralVideoAbiManifest {
    libraries: BTreeMap<String, FixtureGeneralVideoAbiLibrary>,
}

#[derive(Deserialize)]
struct FixtureGeneralVideoAbiLibrary {
    abi_major: u16,
    symbol_version_namespace: String,
    symbols: Vec<String>,
}

#[derive(Clone, Serialize)]
struct PackageManifest {
    schema_version: u16,
    signer: String,
    target: String,
    ffmpeg_release: String,
    source_archive_sha256: String,
    source_signature_sha256: String,
    source_signing_key_fingerprint: String,
    general_abi_sha256: String,
    dependency_contract_sha256: String,
    dependency_contract_receipt_sha256: String,
    license_manifest_sha256: String,
    source_build_manifest_sha256: String,
    libraries: Vec<LibraryManifest>,
    support_files: Vec<String>,
    service_limits: ServiceLimits,
}

#[derive(Clone, Serialize)]
struct LibraryManifest {
    identity: String,
    filename: String,
    sha256: String,
    abi_major: u16,
    soname: String,
    symbol_version_namespace: String,
    symbols: Vec<String>,
    needed: Vec<String>,
}

#[derive(Clone, Serialize)]
struct DependencyManifest {
    schema_version: u16,
    target: String,
    dependencies: Vec<Dependency>,
    edges: Vec<DependencyEdge>,
    encoder_providers: BTreeMap<String, String>,
    reviewed_system_sonames: Vec<String>,
}

#[derive(Clone, Serialize)]
struct Dependency {
    identity: String,
    filename: String,
    sha256: String,
    abi_version: String,
    soname: String,
    needed: Vec<String>,
}

#[derive(Clone, Serialize)]
struct DependencyEdge {
    consumer: String,
    dependency: String,
}

#[derive(Clone, Serialize)]
struct ServiceLimits {
    actor_capacity: u16,
    package_metadata_bytes: u64,
    retained_image_bytes: u64,
    codec_scratch_bytes: u64,
}

#[derive(Serialize)]
struct LicenseManifest {
    schema_version: u16,
    entries: Vec<LicenseEntry>,
}

#[derive(Serialize)]
struct LicenseEntry {
    path: String,
    role: &'static str,
    sha256: String,
}

#[derive(Serialize)]
struct SourceBuildManifest {
    schema_version: u16,
    source_archive_sha256: String,
    source_signature_sha256: String,
    source_signing_key_fingerprint: String,
    source_signature_disposition: &'static str,
    runtime_compilation_forbidden: bool,
    entries: Vec<SourceBuildEntry>,
}

#[derive(Serialize)]
struct SourceBuildEntry {
    path: String,
    role: &'static str,
    sha256: String,
}

#[derive(Serialize)]
struct SignatureReceipt {
    schema_version: u16,
    algorithm: &'static str,
    signature: String,
}

fn main() -> Result<()> {
    let check = std::env::args()
        .skip(1)
        .any(|argument| argument == "--check");
    let fixture = build_fixture()?;
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/video/codec-package-bootstrap");
    if check {
        check_fixture(&root, &fixture)
    } else {
        write_fixture(&root, &fixture)
    }
}

fn build_fixture() -> Result<BTreeMap<String, Vec<u8>>> {
    let abi: FixtureGeneralVideoAbiManifest = serde_json::from_slice(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_runtime/abi/video-codec/ffmpeg-7.1-x86_64-gnu-general-video-v1.json"
    )))?;
    let mut files = BTreeMap::new();
    files.insert(
        "licenses/ffmpeg-license.txt".to_owned(),
        b"FFmpeg 7.1 is distributed under the GNU Lesser General Public License v2.1 or later.\n"
            .to_vec(),
    );
    files.insert(
        "notices/package-notice.txt".to_owned(),
        b"Deterministic reduced package used only for hermetic general-video bootstrap validation.\n"
            .to_vec(),
    );
    files.insert(
        "build/build-provenance.txt".to_owned(),
        b"Synthetic x86_64 ELF images are emitted directly by the tracked Rust fixture generator.\n"
            .to_vec(),
    );
    files.insert(
        "build/build-recipe.txt".to_owned(),
        b"No compiler, subprocess, network, or external package is used.\n".to_vec(),
    );
    files.insert(
        "source/ffmpeg-7.1-source.txt".to_owned(),
        format!("FFmpeg 7.1 archive sha256 {ARCHIVE_SHA256}\n").into_bytes(),
    );

    let dependency_specs = [
        ("svtav1", "lib/libSvtAv1Enc.so.2", "libSvtAv1Enc.so.2", "2"),
        ("vpx", "lib/libvpx.so.9", "libvpx.so.9", "9"),
        ("x264", "lib/libx264.so.164", "libx264.so.164", "164"),
    ];
    let mut dependencies = Vec::new();
    for (identity, filename, soname, abi_version) in dependency_specs {
        let bytes = elf_fixture(62, &[], &["libc.so.6"], soname, "ZED_FIXTURE_1")?;
        let digest = sha256(&bytes);
        files.insert(filename.to_owned(), bytes);
        dependencies.push(Dependency {
            identity: identity.to_owned(),
            filename: filename.to_owned(),
            sha256: digest,
            abi_version: abi_version.to_owned(),
            soname: soname.to_owned(),
            needed: vec!["libc.so.6".to_owned()],
        });
    }

    let mut libraries = Vec::new();
    for (identity, contract) in abi.libraries {
        let (filename, soname) = primary_paths(&identity, contract.abi_major)?;
        let needed = if identity == "avcodec" {
            vec![
                "libSvtAv1Enc.so.2".to_owned(),
                "libc.so.6".to_owned(),
                "libvpx.so.9".to_owned(),
                "libx264.so.164".to_owned(),
            ]
        } else {
            vec!["libc.so.6".to_owned()]
        };
        let needed_refs = needed.iter().map(String::as_str).collect::<Vec<_>>();
        let bytes = elf_fixture(
            62,
            &contract.symbols,
            &needed_refs,
            &soname,
            &contract.symbol_version_namespace,
        )?;
        let digest = sha256(&bytes);
        files.insert(filename.clone(), bytes);
        libraries.push(LibraryManifest {
            identity,
            filename,
            sha256: digest,
            abi_major: contract.abi_major,
            soname,
            symbol_version_namespace: contract.symbol_version_namespace,
            symbols: contract.symbols,
            needed,
        });
    }

    let dependency_manifest = DependencyManifest {
        schema_version: 1,
        target: "x86_64-unknown-linux-gnu".to_owned(),
        dependencies,
        edges: vec![
            DependencyEdge {
                consumer: "avcodec".to_owned(),
                dependency: "svtav1".to_owned(),
            },
            DependencyEdge {
                consumer: "avcodec".to_owned(),
                dependency: "vpx".to_owned(),
            },
            DependencyEdge {
                consumer: "avcodec".to_owned(),
                dependency: "x264".to_owned(),
            },
        ],
        encoder_providers: BTreeMap::from([
            ("aac".to_owned(), "avcodec".to_owned()),
            ("libsvtav1".to_owned(), "svtav1".to_owned()),
            ("libvpx-vp9".to_owned(), "vpx".to_owned()),
            ("libx264".to_owned(), "x264".to_owned()),
        ]),
        reviewed_system_sonames: [
            "libc.so.6",
            "libdl.so.2",
            "libm.so.6",
            "libpthread.so.0",
            "librt.so.1",
        ]
        .map(str::to_owned)
        .to_vec(),
    };
    let dependency_bytes = canonical_json(&dependency_manifest)?;
    let key_pair = fixture_key_pair()?;
    let dependency_receipt = sign_receipt(
        &key_pair,
        DEPENDENCY_DOMAIN,
        FIXTURE_SIGNER,
        &dependency_bytes,
    )?;
    files.insert(
        "dependency-contract-v1.json".to_owned(),
        dependency_bytes.clone(),
    );
    files.insert(
        "dependency-contract-v1.signature.json".to_owned(),
        dependency_receipt.clone(),
    );

    let license_manifest = LicenseManifest {
        schema_version: 1,
        entries: vec![
            disposition_license(&files, "licenses/ffmpeg-license.txt", "license")?,
            disposition_license(&files, "notices/package-notice.txt", "notice")?,
        ],
    };
    let license_bytes = canonical_json(&license_manifest)?;
    files.insert("license-manifest.json".to_owned(), license_bytes.clone());
    let source_build_manifest = SourceBuildManifest {
        schema_version: 1,
        source_archive_sha256: ARCHIVE_SHA256.to_owned(),
        source_signature_sha256: SIGNATURE_SHA256.to_owned(),
        source_signing_key_fingerprint: SIGNING_FINGERPRINT.to_owned(),
        source_signature_disposition: "verified_official_release",
        runtime_compilation_forbidden: true,
        entries: vec![
            disposition_source(&files, "build/build-provenance.txt", "build_provenance")?,
            disposition_source(&files, "build/build-recipe.txt", "build_recipe")?,
            disposition_source(&files, "source/ffmpeg-7.1-source.txt", "source")?,
        ],
    };
    let source_build_bytes = canonical_json(&source_build_manifest)?;
    files.insert(
        "source-build-manifest.json".to_owned(),
        source_build_bytes.clone(),
    );

    let retained_image_bytes = files
        .iter()
        .filter(|(path, _)| path.starts_with("lib/"))
        .try_fold(0_u64, |total, (_, bytes)| {
            total
                .checked_add(u64::try_from(bytes.len())?)
                .ok_or_else(|| anyhow!("image byte accounting overflowed"))
        })?;
    let support_files = [
        "build/build-provenance.txt",
        "build/build-recipe.txt",
        "licenses/ffmpeg-license.txt",
        "notices/package-notice.txt",
        "source/ffmpeg-7.1-source.txt",
    ]
    .map(str::to_owned)
    .to_vec();
    let mut package = PackageManifest {
        schema_version: 1,
        signer: FIXTURE_SIGNER.to_owned(),
        target: "x86_64-unknown-linux-gnu".to_owned(),
        ffmpeg_release: "7.1".to_owned(),
        source_archive_sha256: ARCHIVE_SHA256.to_owned(),
        source_signature_sha256: SIGNATURE_SHA256.to_owned(),
        source_signing_key_fingerprint: SIGNING_FINGERPRINT.to_owned(),
        general_abi_sha256: GENERAL_ABI_SHA256.to_owned(),
        dependency_contract_sha256: sha256(&dependency_bytes),
        dependency_contract_receipt_sha256: sha256(&dependency_receipt),
        license_manifest_sha256: sha256(&license_bytes),
        source_build_manifest_sha256: sha256(&source_build_bytes),
        libraries,
        support_files,
        service_limits: ServiceLimits {
            actor_capacity: 1,
            package_metadata_bytes: 1,
            retained_image_bytes,
            codec_scratch_bytes: CODEC_SCRATCH_BYTES,
        },
    };

    for _ in 0..16 {
        let package_bytes = canonical_json(&package)?;
        files.insert("package-manifest.json".to_owned(), package_bytes);
        let coverage = coverage_bytes(&files)?;
        let receipt = sign_receipt(&key_pair, PACKAGE_DOMAIN, FIXTURE_SIGNER, &coverage)?;
        let metadata_bytes = u64::try_from(coverage.len())?
            .checked_add(u64::try_from(receipt.len())?)
            .and_then(|total| {
                files
                    .iter()
                    .filter(|(path, _)| !path.starts_with("lib/"))
                    .try_fold(total, |total, (_, bytes)| {
                        total.checked_add(u64::try_from(bytes.len()).ok()?)
                    })
            })
            .ok_or_else(|| anyhow!("metadata byte accounting overflowed"))?;
        if package.service_limits.package_metadata_bytes == metadata_bytes {
            files.insert("package-coverage.sha256".to_owned(), coverage);
            files.insert("package-signature.json".to_owned(), receipt);
            return Ok(files);
        }
        package.service_limits.package_metadata_bytes = metadata_bytes;
    }
    bail!("package metadata byte accounting did not converge")
}

fn disposition_license(
    files: &BTreeMap<String, Vec<u8>>,
    path: &str,
    role: &'static str,
) -> Result<LicenseEntry> {
    Ok(LicenseEntry {
        path: path.to_owned(),
        role,
        sha256: sha256(required_file(files, path)?),
    })
}

fn disposition_source(
    files: &BTreeMap<String, Vec<u8>>,
    path: &str,
    role: &'static str,
) -> Result<SourceBuildEntry> {
    Ok(SourceBuildEntry {
        path: path.to_owned(),
        role,
        sha256: sha256(required_file(files, path)?),
    })
}

fn required_file<'a>(files: &'a BTreeMap<String, Vec<u8>>, path: &str) -> Result<&'a [u8]> {
    files
        .get(path)
        .map(Vec::as_slice)
        .ok_or_else(|| anyhow!("fixture file {path} is absent"))
}

fn primary_paths(identity: &str, major: u16) -> Result<(String, String)> {
    let soname = match identity {
        "avcodec" | "avfilter" | "avformat" | "avutil" | "swresample" | "swscale" => {
            format!("lib{identity}.so.{major}")
        }
        _ => bail!("unsupported ABI library identity {identity}"),
    };
    Ok((format!("lib/{soname}"), soname))
}

fn canonical_json(value: &impl Serialize) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn fixture_key_pair() -> Result<Ed25519KeyPair> {
    let seed = decode_hex::<32>(FIXTURE_SEED_HEX)?;
    let pair = Ed25519KeyPair::from_seed_unchecked(&seed)
        .map_err(|_| anyhow!("fixture Ed25519 seed is invalid"))?;
    if encode_hex(pair.public_key().as_ref())
        != "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
    {
        bail!("fixture Ed25519 public key changed")
    }
    Ok(pair)
}

fn sign_receipt(
    key_pair: &Ed25519KeyPair,
    domain: &[u8],
    signer: &str,
    payload: &[u8],
) -> Result<Vec<u8>> {
    let signing_payload = signing_payload(domain, signer, payload)?;
    canonical_json(&SignatureReceipt {
        schema_version: 1,
        algorithm: "ed25519",
        signature: encode_hex(key_pair.sign(&signing_payload).as_ref()),
    })
}

fn signing_payload(domain: &[u8], signer: &str, payload: &[u8]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(
        domain
            .len()
            .checked_add(8)
            .and_then(|length| length.checked_add(signer.len()))
            .and_then(|length| length.checked_add(8))
            .and_then(|length| length.checked_add(payload.len()))
            .ok_or_else(|| anyhow!("signing payload length overflowed"))?,
    )?;
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&u64::try_from(signer.len())?.to_be_bytes());
    bytes.extend_from_slice(signer.as_bytes());
    bytes.extend_from_slice(&u64::try_from(payload.len())?.to_be_bytes());
    bytes.extend_from_slice(payload);
    Ok(bytes)
}

fn coverage_bytes(files: &BTreeMap<String, Vec<u8>>) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for (path, contents) in files {
        if path == "package-coverage.sha256" || path == "package-signature.json" {
            continue;
        }
        bytes.extend_from_slice(
            format!("{} {}  {path}\n", sha256(contents), contents.len()).as_bytes(),
        );
    }
    Ok(bytes)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn decode_hex<const N: usize>(value: &str) -> Result<[u8; N]> {
    if value.len() != N * 2 {
        bail!("hex input has the wrong length")
    }
    let mut output = [0_u8; N];
    for (slot, pair) in output.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
        *slot = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Ok(output)
}

fn hex_nibble(value: u8) -> Result<u8> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => bail!("hex input is not lowercase hexadecimal"),
    }
}

fn elf_fixture(
    machine: u16,
    symbols: &[String],
    needed: &[&str],
    soname: &str,
    version_namespace: &str,
) -> Result<Vec<u8>> {
    let mut strings = vec![0_u8];
    let mut names = Vec::new();
    for symbol in symbols {
        names.push(u32::try_from(strings.len())?);
        strings.extend_from_slice(symbol.as_bytes());
        strings.push(0);
    }
    let mut needed_offsets = Vec::new();
    for dependency in needed {
        needed_offsets.push(u64::try_from(strings.len())?);
        strings.extend_from_slice(dependency.as_bytes());
        strings.push(0);
    }
    let soname_offset = u64::try_from(strings.len())?;
    strings.extend_from_slice(soname.as_bytes());
    strings.push(0);
    let version_name_offset = u32::try_from(strings.len())?;
    strings.extend_from_slice(version_namespace.as_bytes());
    strings.push(0);

    let program_offset = 64_usize;
    let program_entry_size = 56_usize;
    let program_count = 2_usize;
    let string_offset = 192_usize;
    let symbol_offset = align(string_offset + strings.len(), 8)?;
    let symbol_count = symbols
        .len()
        .checked_add(1)
        .ok_or_else(|| anyhow!("symbol count overflowed"))?;
    let symbol_size = symbol_count
        .checked_mul(24)
        .ok_or_else(|| anyhow!("symbol table overflowed"))?;
    let version_symbol_offset = symbol_offset
        .checked_add(symbol_size)
        .ok_or_else(|| anyhow!("ELF layout overflowed"))?;
    let version_symbol_size = symbol_count
        .checked_mul(2)
        .ok_or_else(|| anyhow!("version table overflowed"))?;
    let version_definition_offset = align(version_symbol_offset + version_symbol_size, 4)?;
    let version_definition_size = 56_usize;
    let hash_offset = align(version_definition_offset + version_definition_size, 4)?;
    let hash_words = 2_usize
        .checked_add(1)
        .and_then(|value| value.checked_add(symbol_count))
        .ok_or_else(|| anyhow!("hash table overflowed"))?;
    let hash_size = hash_words
        .checked_mul(4)
        .ok_or_else(|| anyhow!("hash table overflowed"))?;
    let code_offset = align(hash_offset + hash_size, 16)?;
    let code_size = symbols
        .len()
        .checked_mul(16)
        .ok_or_else(|| anyhow!("code table overflowed"))?;
    let dynamic_offset = align(code_offset + code_size, 8)?;
    let dynamic_entries = 10_usize
        .checked_add(needed_offsets.len())
        .ok_or_else(|| anyhow!("dynamic table overflowed"))?;
    let dynamic_size = dynamic_entries
        .checked_mul(16)
        .ok_or_else(|| anyhow!("dynamic table overflowed"))?;
    let section_offset = align(dynamic_offset + dynamic_size, 8)?;
    let byte_count = section_offset
        .checked_add(6 * 64)
        .ok_or_else(|| anyhow!("ELF size overflowed"))?;
    let mut bytes = vec![0_u8; byte_count];
    bytes[..4].copy_from_slice(b"\x7fELF");
    bytes[4] = 2;
    bytes[5] = 1;
    bytes[6] = 1;
    write_u16(&mut bytes, 16, 3)?;
    write_u16(&mut bytes, 18, machine)?;
    write_u32(&mut bytes, 20, 1)?;
    write_u64(&mut bytes, 32, u64::try_from(program_offset)?)?;
    write_u64(&mut bytes, 40, u64::try_from(section_offset)?)?;
    write_u16(&mut bytes, 52, 64)?;
    write_u16(&mut bytes, 54, u16::try_from(program_entry_size)?)?;
    write_u16(&mut bytes, 56, u16::try_from(program_count)?)?;
    write_u16(&mut bytes, 58, 64)?;
    write_u16(&mut bytes, 60, 6)?;
    let file_length = u64::try_from(bytes.len())?;
    write_u32(&mut bytes, program_offset, 1)?;
    write_u32(&mut bytes, program_offset + 4, 5)?;
    write_u64(&mut bytes, program_offset + 32, file_length)?;
    write_u64(&mut bytes, program_offset + 40, file_length)?;
    write_u64(&mut bytes, program_offset + 48, 0x1000)?;
    let dynamic_program = program_offset + program_entry_size;
    write_u32(&mut bytes, dynamic_program, 2)?;
    write_u32(&mut bytes, dynamic_program + 4, 4)?;
    write_u64(
        &mut bytes,
        dynamic_program + 8,
        u64::try_from(dynamic_offset)?,
    )?;
    write_u64(
        &mut bytes,
        dynamic_program + 16,
        u64::try_from(dynamic_offset)?,
    )?;
    write_u64(
        &mut bytes,
        dynamic_program + 32,
        u64::try_from(dynamic_size)?,
    )?;
    write_u64(
        &mut bytes,
        dynamic_program + 40,
        u64::try_from(dynamic_size)?,
    )?;
    write_u64(&mut bytes, dynamic_program + 48, 8)?;
    bytes[string_offset..string_offset + strings.len()].copy_from_slice(&strings);

    for (index, (name_offset, symbol)) in names.iter().zip(symbols).enumerate() {
        let entry = symbol_offset + (index + 1) * 24;
        write_u32(&mut bytes, entry, *name_offset)?;
        bytes[entry + 4] = 0x12;
        write_u16(&mut bytes, entry + 6, 1)?;
        let function_offset = code_offset + index * 16;
        write_u64(&mut bytes, entry + 8, u64::try_from(function_offset)?)?;
        write_u64(&mut bytes, entry + 16, 16)?;
        write_u16(&mut bytes, version_symbol_offset + (index + 1) * 2, 2)?;
        let code = function_code(symbol);
        let destination = bytes
            .get_mut(function_offset..function_offset + code.len())
            .ok_or_else(|| anyhow!("function code exceeds ELF bounds"))?;
        destination.copy_from_slice(&code);
    }

    write_u16(&mut bytes, version_definition_offset, 1)?;
    write_u16(&mut bytes, version_definition_offset + 2, 1)?;
    write_u16(&mut bytes, version_definition_offset + 4, 1)?;
    write_u16(&mut bytes, version_definition_offset + 6, 1)?;
    write_u32(&mut bytes, version_definition_offset + 12, 20)?;
    write_u32(&mut bytes, version_definition_offset + 16, 28)?;
    write_u32(
        &mut bytes,
        version_definition_offset + 20,
        u32::try_from(soname_offset)?,
    )?;
    let callable_definition = version_definition_offset + 28;
    write_u16(&mut bytes, callable_definition, 1)?;
    write_u16(&mut bytes, callable_definition + 4, 2)?;
    write_u16(&mut bytes, callable_definition + 6, 1)?;
    write_u32(&mut bytes, callable_definition + 12, 20)?;
    write_u32(&mut bytes, callable_definition + 20, version_name_offset)?;

    write_u32(&mut bytes, hash_offset, 1)?;
    write_u32(&mut bytes, hash_offset + 4, u32::try_from(symbol_count)?)?;
    write_u32(&mut bytes, hash_offset + 8, u32::from(!symbols.is_empty()))?;
    for index in 1..symbol_count {
        let next = if index + 1 < symbol_count {
            index + 1
        } else {
            0
        };
        write_u32(
            &mut bytes,
            hash_offset + 12 + index * 4,
            u32::try_from(next)?,
        )?;
    }

    let dynamic_values = [
        (4_u64, u64::try_from(hash_offset)?),
        (5, u64::try_from(string_offset)?),
        (10, u64::try_from(strings.len())?),
        (6, u64::try_from(symbol_offset)?),
        (11, 24),
        (14, soname_offset),
        (0x6fff_fff0, u64::try_from(version_symbol_offset)?),
        (0x6fff_fffc, u64::try_from(version_definition_offset)?),
        (0x6fff_fffd, 2),
    ];
    let mut dynamic_index = 0_usize;
    for (tag, value) in dynamic_values {
        write_u64(&mut bytes, dynamic_offset + dynamic_index * 16, tag)?;
        write_u64(&mut bytes, dynamic_offset + dynamic_index * 16 + 8, value)?;
        dynamic_index += 1;
    }
    for offset in needed_offsets {
        write_u64(&mut bytes, dynamic_offset + dynamic_index * 16, 1)?;
        write_u64(&mut bytes, dynamic_offset + dynamic_index * 16 + 8, offset)?;
        dynamic_index += 1;
    }
    write_u64(&mut bytes, dynamic_offset + dynamic_index * 16, 0)?;

    let strings_header = section_offset + 64;
    write_u32(&mut bytes, strings_header + 4, 3)?;
    write_u64(
        &mut bytes,
        strings_header + 16,
        u64::try_from(string_offset)?,
    )?;
    write_u64(
        &mut bytes,
        strings_header + 24,
        u64::try_from(string_offset)?,
    )?;
    write_u64(
        &mut bytes,
        strings_header + 32,
        u64::try_from(strings.len())?,
    )?;
    let symbols_header = section_offset + 128;
    write_u32(&mut bytes, symbols_header + 4, 11)?;
    write_u64(&mut bytes, symbols_header + 8, 2)?;
    write_u64(
        &mut bytes,
        symbols_header + 16,
        u64::try_from(symbol_offset)?,
    )?;
    write_u64(
        &mut bytes,
        symbols_header + 24,
        u64::try_from(symbol_offset)?,
    )?;
    write_u64(&mut bytes, symbols_header + 32, u64::try_from(symbol_size)?)?;
    write_u32(&mut bytes, symbols_header + 40, 1)?;
    write_u64(&mut bytes, symbols_header + 56, 24)?;
    let dynamic_header = section_offset + 192;
    write_u32(&mut bytes, dynamic_header + 4, 6)?;
    write_u64(
        &mut bytes,
        dynamic_header + 16,
        u64::try_from(dynamic_offset)?,
    )?;
    write_u64(
        &mut bytes,
        dynamic_header + 24,
        u64::try_from(dynamic_offset)?,
    )?;
    write_u64(
        &mut bytes,
        dynamic_header + 32,
        u64::try_from(dynamic_size)?,
    )?;
    write_u32(&mut bytes, dynamic_header + 40, 1)?;
    write_u64(&mut bytes, dynamic_header + 56, 16)?;
    let version_symbol_header = section_offset + 256;
    write_u32(&mut bytes, version_symbol_header + 4, 0x6fff_ffff)?;
    write_u64(&mut bytes, version_symbol_header + 8, 2)?;
    write_u64(
        &mut bytes,
        version_symbol_header + 16,
        u64::try_from(version_symbol_offset)?,
    )?;
    write_u64(
        &mut bytes,
        version_symbol_header + 24,
        u64::try_from(version_symbol_offset)?,
    )?;
    write_u64(
        &mut bytes,
        version_symbol_header + 32,
        u64::try_from(version_symbol_size)?,
    )?;
    write_u32(&mut bytes, version_symbol_header + 40, 2)?;
    write_u64(&mut bytes, version_symbol_header + 56, 2)?;
    let version_definition_header = section_offset + 320;
    write_u32(&mut bytes, version_definition_header + 4, 0x6fff_fffd)?;
    write_u64(&mut bytes, version_definition_header + 8, 2)?;
    write_u64(
        &mut bytes,
        version_definition_header + 16,
        u64::try_from(version_definition_offset)?,
    )?;
    write_u64(
        &mut bytes,
        version_definition_header + 24,
        u64::try_from(version_definition_offset)?,
    )?;
    write_u64(
        &mut bytes,
        version_definition_header + 32,
        u64::try_from(version_definition_size)?,
    )?;
    write_u32(&mut bytes, version_definition_header + 40, 1)?;
    write_u32(&mut bytes, version_definition_header + 44, 2)?;
    Ok(bytes)
}

fn function_code(symbol: &str) -> Vec<u8> {
    let version = match symbol {
        "avcodec_version" => Some(0x3d1364_u32),
        "avformat_version" => Some(0x3d0764),
        "avutil_version" => Some(0x3b2764),
        "swresample_version" => Some(0x050364),
        "swscale_version" => Some(0x080364),
        "avfilter_version" => Some(0x0a0464),
        _ => None,
    };
    if let Some(version) = version {
        let mut code = vec![0xb8];
        code.extend_from_slice(&version.to_le_bytes());
        code.push(0xc3);
        return code;
    }
    if matches!(
        symbol,
        "avcodec_find_encoder_by_name" | "avcodec_find_decoder"
    ) {
        return vec![0xb8, 1, 0, 0, 0, 0xc3];
    }
    vec![0x31, 0xc0, 0xc3]
}

fn align(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| anyhow!("ELF alignment overflowed"))
}

fn write_u16(bytes: &mut [u8], offset: usize, value: u16) -> Result<()> {
    bytes
        .get_mut(offset..offset + 2)
        .ok_or_else(|| anyhow!("ELF u16 write exceeds bounds"))?
        .copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) -> Result<()> {
    bytes
        .get_mut(offset..offset + 4)
        .ok_or_else(|| anyhow!("ELF u32 write exceeds bounds"))?
        .copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) -> Result<()> {
    bytes
        .get_mut(offset..offset + 8)
        .ok_or_else(|| anyhow!("ELF u64 write exceeds bounds"))?
        .copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_fixture(root: &Path, fixture: &BTreeMap<String, Vec<u8>>) -> Result<()> {
    for (path, bytes) in fixture {
        let destination = root.join(path);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(destination, bytes)?;
    }
    check_fixture(root, fixture)
}

fn check_fixture(root: &Path, fixture: &BTreeMap<String, Vec<u8>>) -> Result<()> {
    let actual_paths = recursive_files(root)?;
    let expected_paths = fixture.keys().cloned().collect::<BTreeSet<_>>();
    if actual_paths != expected_paths {
        bail!("tracked fixture file set differs from deterministic output")
    }
    for (path, expected) in fixture {
        let actual = fs::read(root.join(path)).with_context(|| format!("read fixture {path}"))?;
        if &actual != expected {
            bail!("tracked fixture {path} differs from deterministic output")
        }
    }
    Ok(())
}

fn recursive_files(root: &Path) -> Result<BTreeSet<String>> {
    fn visit(root: &Path, current: &Path, output: &mut BTreeSet<String>) -> Result<()> {
        for entry in fs::read_dir(current)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                visit(root, &path, output)?;
            } else if file_type.is_file() {
                let relative = path
                    .strip_prefix(root)?
                    .to_str()
                    .ok_or_else(|| anyhow!("fixture path is not UTF-8"))?
                    .replace(std::path::MAIN_SEPARATOR, "/");
                output.insert(relative);
            } else {
                bail!("fixture contains a non-regular entry")
            }
        }
        Ok(())
    }
    let mut output = BTreeSet::new();
    if root.exists() {
        visit(root, root, &mut output)?;
    }
    Ok(output)
}
