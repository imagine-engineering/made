use std::{
    collections::BTreeSet,
    env,
    error::Error,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

const EVIDENCE_PATH: &str = "abi/reviewed-bindings-v1.txt";
const MANIFEST_PATH: &str = "abi/symbols-v1.json";
const CATALOG_PATH: &str = "../../.agents/specs/comfy-parity/catalogs/native-backend-abi/rocm.json";
const REVIEWED_HEADER_DIRECTORY_ENV: &str = "SIM_ROCM_REVIEWED_HEADER_DIR";
const REQUIRE_COMPLETION_EVIDENCE_ENV: &str = "SIM_ROCM_REQUIRE_COMPLETION_EVIDENCE";
const COMPLETION_EVIDENCE_OUTPUT_ENV: &str = "SIM_ROCM_COMPLETION_EVIDENCE_OUT";
const COMPLETION_EVIDENCE_RUN_ID_ENV: &str = "SIM_ROCM_COMPLETION_EVIDENCE_RUN_ID";

const EXPECTED_HEADERS: [(&str, &str, usize, &str); 9] = [
    (
        "hip_runtime_api.h",
        "https://raw.githubusercontent.com/ROCm/HIP/rocm-6.1.2/include/hip/hip_runtime_api.h",
        386_498,
        "7f36ffc64b62b0255ca76a561be64a8481be6154330e31b23ede30bc0e226b23",
    ),
    (
        "hip_common.h",
        "https://raw.githubusercontent.com/ROCm/HIP/rocm-6.1.2/include/hip/hip_common.h",
        3_450,
        "ab968f846dc31d6d22509edac92e657ccfd99d940d6cc98b99f74794ec1b3dd0",
    ),
    (
        "driver_types.h",
        "https://raw.githubusercontent.com/ROCm/HIP/rocm-6.1.2/include/hip/driver_types.h",
        18_985,
        "9b51c8f341c2f34aa44de728d89774ef0a0bfe34f2e3c973f49883095ffccec8",
    ),
    (
        "hiprtc.h",
        "https://raw.githubusercontent.com/ROCm/HIP/rocm-6.1.2/include/hip/hiprtc.h",
        15_631,
        "9e92ba7f66646087b96e52481aea1fb931b43eb48c4b69a414152aadd3a9d00d",
    ),
    (
        "rocblas.h",
        "https://raw.githubusercontent.com/ROCm/rocBLAS/rocm-6.1.2/library/include/rocblas.h",
        1_748,
        "312b39e9a670a780abe9a87414624b60d54cc52c019b34567e8a063348cca577",
    ),
    (
        "rocblas-auxiliary.h",
        "https://raw.githubusercontent.com/ROCm/rocBLAS/rocm-6.1.2/library/include/internal/rocblas-auxiliary.h",
        18_040,
        "2d576576c044b6ce36a2e2ffeb8924b0004e622dca5d992802fdc130b7f7d217",
    ),
    (
        "rocblas-functions.h",
        "https://raw.githubusercontent.com/ROCm/rocBLAS/rocm-6.1.2/library/include/internal/rocblas-functions.h",
        1_087_850,
        "30dd7c36decde74904a89f77d42ed8d239b201ff5fbae45b2315347c4fc5f337",
    ),
    (
        "rocblas-types.h",
        "https://raw.githubusercontent.com/ROCm/rocBLAS/rocm-6.1.2/library/include/internal/rocblas-types.h",
        14_718,
        "d7fa26470c43dabdf5786b6ece86db62b5f0ced1ded227e805ec4373879ab305",
    ),
    (
        "miopen.h",
        "https://raw.githubusercontent.com/ROCm/MIOpen/rocm-6.1.2/include/miopen/miopen.h",
        292_299,
        "95f4ee132da9da5c53548331b445d7faf207bfbb13855de2eab57a9bae5438a9",
    ),
];

#[derive(Debug)]
struct Header {
    name: String,
    source_url: String,
    byte_length: usize,
    digest: String,
}

#[derive(Debug)]
struct Symbol {
    library: String,
    name: String,
    alias: String,
    header: String,
    line_start: usize,
    line_end: usize,
    digest: String,
    declaration: String,
}

#[derive(Debug)]
struct Layout {
    c_name: String,
    rust_name: String,
    header: String,
    line_start: usize,
    line_end: usize,
    digest: String,
    size: usize,
    align: usize,
    declaration: String,
}

#[derive(Debug)]
struct ConstantSet {
    name: String,
    header: String,
    line_start: usize,
    line_end: usize,
    digest: String,
}

#[derive(Clone, Copy)]
struct ConstantSpec {
    c_name: &'static str,
    rust_name: &'static str,
    rust_type: &'static str,
    value: i32,
}

fn constant_specs(name: &str) -> Option<&'static [ConstantSpec]> {
    match name {
        "hipError" => Some(&[
            ConstantSpec {
                c_name: "hipSuccess",
                rust_name: "HIP_SUCCESS",
                rust_type: "HipError",
                value: 0,
            },
            ConstantSpec {
                c_name: "hipErrorOutOfMemory",
                rust_name: "HIP_ERROR_OUT_OF_MEMORY",
                rust_type: "HipError",
                value: 2,
            },
            ConstantSpec {
                c_name: "hipErrorInvalidContext",
                rust_name: "HIP_ERROR_INVALID_CONTEXT",
                rust_type: "HipError",
                value: 201,
            },
            ConstantSpec {
                c_name: "hipErrorIllegalAddress",
                rust_name: "HIP_ERROR_ILLEGAL_ADDRESS",
                rust_type: "HipError",
                value: 700,
            },
            ConstantSpec {
                c_name: "hipErrorContextIsDestroyed",
                rust_name: "HIP_ERROR_CONTEXT_IS_DESTROYED",
                rust_type: "HipError",
                value: 709,
            },
            ConstantSpec {
                c_name: "hipErrorLaunchFailure",
                rust_name: "HIP_ERROR_LAUNCH_FAILURE",
                rust_type: "HipError",
                value: 719,
            },
        ]),
        "hipExecutionFlags" => Some(&[
            ConstantSpec {
                c_name: "hipStreamNonBlocking",
                rust_name: "HIP_STREAM_NON_BLOCKING",
                rust_type: "HipStreamFlags",
                value: 1,
            },
            ConstantSpec {
                c_name: "hipEventDisableTiming",
                rust_name: "HIP_EVENT_DISABLE_TIMING",
                rust_type: "HipEventFlags",
                value: 2,
            },
        ]),
        "hipDeviceAttributeComputeCapability" => Some(&[
            ConstantSpec {
                c_name: "hipDeviceAttributeComputeCapabilityMajor",
                rust_name: "HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR",
                rust_type: "HipDeviceAttribute",
                value: 23,
            },
            ConstantSpec {
                c_name: "hipDeviceAttributeComputeCapabilityMinor",
                rust_name: "HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR",
                rust_type: "HipDeviceAttribute",
                value: 61,
            },
        ]),
        "hipInitFlags" => Some(&[ConstantSpec {
            c_name: "SIM_HIP_INIT_FLAGS_ZERO",
            rust_name: "HIP_INIT_FLAGS_ZERO",
            rust_type: "HipInitFlags",
            value: 0,
        }]),
        "hipMemcpyKind" => Some(&[
            ConstantSpec {
                c_name: "hipMemcpyHostToDevice",
                rust_name: "HIP_MEMCPY_HOST_TO_DEVICE",
                rust_type: "HipMemcpyKind",
                value: 1,
            },
            ConstantSpec {
                c_name: "hipMemcpyDeviceToHost",
                rust_name: "HIP_MEMCPY_DEVICE_TO_HOST",
                rust_type: "HipMemcpyKind",
                value: 2,
            },
            ConstantSpec {
                c_name: "hipMemcpyDeviceToDevice",
                rust_name: "HIP_MEMCPY_DEVICE_TO_DEVICE",
                rust_type: "HipMemcpyKind",
                value: 3,
            },
        ]),
        "rocblasOperation" => Some(&[ConstantSpec {
            c_name: "rocblas_operation_none",
            rust_name: "ROCBLAS_OPERATION_NONE",
            rust_type: "RocblasOperation",
            value: 111,
        }]),
        "rocblasStatus" => Some(&[
            ConstantSpec {
                c_name: "rocblas_status_success",
                rust_name: "ROCBLAS_STATUS_SUCCESS",
                rust_type: "RocblasStatus",
                value: 0,
            },
            ConstantSpec {
                c_name: "rocblas_status_memory_error",
                rust_name: "ROCBLAS_STATUS_MEMORY_ERROR",
                rust_type: "RocblasStatus",
                value: 5,
            },
        ]),
        "hiprtcResult" => Some(&[ConstantSpec {
            c_name: "HIPRTC_SUCCESS",
            rust_name: "HIPRTC_SUCCESS",
            rust_type: "HipRtcResult",
            value: 0,
        }]),
        "miopenStatus" => Some(&[ConstantSpec {
            c_name: "miopenStatusSuccess",
            rust_name: "MIOPEN_STATUS_SUCCESS",
            rust_type: "MiopenStatus",
            value: 0,
        }]),
        _ => None,
    }
}

#[derive(Debug)]
struct CompletionEvidence {
    artifact_sha256: String,
    run_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LayoutMeasurement {
    name: String,
    size: usize,
    align: usize,
}

#[derive(Debug)]
struct ProbeEvidence {
    c_compiler_version: String,
    rustc_version: String,
    c_source_sha256: String,
    cross_object_sha256: String,
    rust_source_sha256: String,
    c_measurements: Vec<LayoutMeasurement>,
    rust_measurements: Vec<LayoutMeasurement>,
    cross_compile_arguments: Vec<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    const INPUTS: [&str; 5] = [
        EVIDENCE_PATH,
        MANIFEST_PATH,
        "LICENSES",
        CATALOG_PATH,
        "src/abi.rs",
    ];

    for input in INPUTS {
        println!("cargo:rerun-if-changed={input}");
        if !Path::new(input).is_file() {
            return Err(format!("required checked ROCm ABI input is missing: {input}").into());
        }
    }
    for variable in [
        REVIEWED_HEADER_DIRECTORY_ENV,
        REQUIRE_COMPLETION_EVIDENCE_ENV,
        COMPLETION_EVIDENCE_OUTPUT_ENV,
        COMPLETION_EVIDENCE_RUN_ID_ENV,
        "CC",
        "RUSTC",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    if sha256_hex(b"abc") != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" {
        return Err("internal SHA-256 implementation failed its known-answer test".into());
    }

    let evidence = fs::read_to_string(EVIDENCE_PATH)?;
    let manifest = fs::read_to_string(MANIFEST_PATH)?;
    let catalog = fs::read_to_string(CATALOG_PATH)?;
    let (headers, symbols, layouts, constant_sets) = parse_evidence(&evidence)?;
    validate_headers(&headers, &manifest, &catalog)?;
    validate_unique_records(&symbols, &layouts, &constant_sets)?;
    let output_directory = PathBuf::from(env::var("OUT_DIR")?);
    let completion = completion_evidence(
        &headers,
        &symbols,
        &layouts,
        &constant_sets,
        &evidence,
        &output_directory,
    )?;
    println!(
        "cargo:rustc-env=SIM_ROCM_COMPLETION_EVIDENCE_STATE={}",
        if completion.is_some() {
            "verified"
        } else {
            "unavailable"
        }
    );
    let generated = generate_bindings(
        &headers,
        &symbols,
        &layouts,
        &constant_sets,
        &manifest,
        completion.as_ref(),
    )?;
    let output = output_directory.join("rocm_abi_bindings.rs");
    fs::write(output, generated)?;

    let target = env::var("TARGET").unwrap_or_else(|_| "unknown-target".to_owned());
    println!("cargo:rustc-env=SIM_ROCM_BUILD_TARGET={target}");
    Ok(())
}

fn completion_evidence(
    headers: &[Header],
    symbols: &[Symbol],
    layouts: &[Layout],
    constant_sets: &[ConstantSet],
    ledger: &str,
    output_directory: &Path,
) -> Result<Option<CompletionEvidence>, String> {
    let requested = match env::var(REQUIRE_COMPLETION_EVIDENCE_ENV) {
        Ok(value) if value == "1" => true,
        Ok(value) => {
            return Err(format!(
                "{REQUIRE_COMPLETION_EVIDENCE_ENV} must be exactly 1 when set, found {value:?}"
            ));
        }
        Err(env::VarError::NotPresent) => false,
        Err(error) => {
            return Err(format!(
                "failed to read {REQUIRE_COMPLETION_EVIDENCE_ENV}: {error}"
            ));
        }
    };

    if !requested {
        if let Some(directory) = env::var_os(REVIEWED_HEADER_DIRECTORY_ENV) {
            validate_full_header_bytes(
                Path::new(&directory),
                headers,
                symbols,
                layouts,
                constant_sets,
            )?;
        }
        return Ok(None);
    }

    let header_directory = required_path(REVIEWED_HEADER_DIRECTORY_ENV)?;
    let artifact_path = required_path(COMPLETION_EVIDENCE_OUTPUT_ENV)?;
    let run_id = required_string(COMPLETION_EVIDENCE_RUN_ID_ENV)?;
    validate_full_header_bytes(&header_directory, headers, symbols, layouts, constant_sets)?;
    validate_completion_layout_set(layouts)?;

    let compiler = env::var_os("CC").unwrap_or_else(|| OsString::from("cc"));
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"));
    let probe = run_layout_probes(
        layouts,
        constant_sets,
        &header_directory,
        output_directory,
        &compiler,
        &rustc,
    )?;
    let artifact = render_completion_artifact(
        headers,
        symbols,
        layouts,
        constant_sets,
        ledger,
        &run_id,
        &compiler,
        &rustc,
        &probe,
    );
    if let Some(parent) = artifact_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create completion-evidence directory {}: {error}",
                parent.display()
            )
        })?;
    }
    fs::write(&artifact_path, artifact.as_bytes()).map_err(|error| {
        format!(
            "failed to write completion evidence {}: {error}",
            artifact_path.display()
        )
    })?;
    Ok(Some(CompletionEvidence {
        artifact_sha256: sha256_hex(artifact.as_bytes()),
        run_id,
    }))
}

fn required_path(variable: &str) -> Result<PathBuf, String> {
    env::var_os(variable)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| {
            format!(
                "completion evidence requires {variable}; ordinary offline builds remain available without setting {REQUIRE_COMPLETION_EVIDENCE_ENV}=1"
            )
        })
}

fn required_string(variable: &str) -> Result<String, String> {
    env::var(variable)
        .ok()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("completion evidence requires non-empty {variable}"))
}

fn validate_completion_layout_set(layouts: &[Layout]) -> Result<(), String> {
    let expected = ["hipUUID", "hipIpcMemHandle_t", "miopenConvAlgoPerf_t"];
    for name in expected {
        if layouts
            .iter()
            .filter(|layout| layout.c_name == name)
            .count()
            != 1
        {
            return Err(format!(
                "completion evidence requires exactly one compiled C layout proof for {name}"
            ));
        }
    }
    Ok(())
}

fn validate_full_header_bytes(
    directory: &Path,
    headers: &[Header],
    symbols: &[Symbol],
    layouts: &[Layout],
    constant_sets: &[ConstantSet],
) -> Result<(), String> {
    for header in headers {
        let path = directory.join(&header.name);
        let bytes = fs::read(&path).map_err(|error| {
            format!("failed to read reviewed header {}: {error}", path.display())
        })?;
        if bytes.len() != header.byte_length || sha256_hex(&bytes) != header.digest {
            return Err(format!(
                "reviewed header {} does not match the pinned ROCm 6.1.2 byte length and SHA-256",
                path.display()
            ));
        }
        let text = std::str::from_utf8(&bytes)
            .map_err(|error| format!("reviewed header {} is not UTF-8: {error}", path.display()))?;
        validate_header_ranges(&header.name, &path, text, symbols, layouts, constant_sets)?;
    }
    Ok(())
}

fn validate_header_ranges(
    header_name: &str,
    path: &Path,
    text: &str,
    symbols: &[Symbol],
    layouts: &[Layout],
    constant_sets: &[ConstantSet],
) -> Result<(), String> {
    for (kind, name, start, end, declaration) in symbols
        .iter()
        .filter(|symbol| symbol.header == header_name)
        .map(|symbol| {
            (
                "symbol",
                symbol.name.as_str(),
                symbol.line_start,
                symbol.line_end,
                symbol.declaration.as_str(),
            )
        })
        .chain(
            layouts
                .iter()
                .filter(|layout| layout.header == header_name)
                .map(|layout| {
                    (
                        "layout",
                        layout.c_name.as_str(),
                        layout.line_start,
                        layout.line_end,
                        layout.declaration.as_str(),
                    )
                }),
        )
    {
        let exact_bytes = text
            .lines()
            .skip(start.saturating_sub(1))
            .take(end.saturating_sub(start) + 1)
            .collect::<Vec<_>>()
            .join("\n");
        if exact_bytes != declaration {
            return Err(format!(
                "reviewed {kind} {name} is not the exact byte range {start}-{end} in {}",
                path.display()
            ));
        }
    }
    for constant_set in constant_sets
        .iter()
        .filter(|constant_set| constant_set.header == header_name)
    {
        let exact_bytes = text
            .lines()
            .skip(constant_set.line_start.saturating_sub(1))
            .take(
                constant_set
                    .line_end
                    .saturating_sub(constant_set.line_start)
                    + 1,
            )
            .collect::<Vec<_>>()
            .join("\n");
        let names_are_bound = if constant_set.name == "hipInitFlags" {
            exact_bytes.contains("flags  Initialization flag, should be zero")
        } else {
            constant_specs(&constant_set.name).is_some_and(|constants| {
                constants
                    .iter()
                    .all(|constant| exact_bytes.contains(constant.c_name))
            })
        };
        if sha256_hex(exact_bytes.as_bytes()) != constant_set.digest || !names_are_bound {
            return Err(format!(
                "reviewed constant set {} is not the exact byte range {}-{} in {}",
                constant_set.name,
                constant_set.line_start,
                constant_set.line_end,
                path.display()
            ));
        }
    }
    Ok(())
}

fn run_layout_probes(
    layouts: &[Layout],
    constant_sets: &[ConstantSet],
    header_directory: &Path,
    output_directory: &Path,
    compiler: &OsString,
    rustc: &OsString,
) -> Result<ProbeEvidence, String> {
    let constant_declarations = constant_sets
        .iter()
        .map(|constant_set| {
            let text = fs::read_to_string(header_directory.join(&constant_set.header))
                .map_err(|error| format!("failed to read constant-set header: {error}"))?;
            Ok((
                constant_set.name.clone(),
                text.lines()
                    .skip(constant_set.line_start.saturating_sub(1))
                    .take(
                        constant_set
                            .line_end
                            .saturating_sub(constant_set.line_start)
                            + 1,
                    )
                    .collect::<Vec<_>>()
                    .join("\n"),
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let c_source = render_c_layout_probe(layouts, &constant_declarations);
    let c_source_path = output_directory.join("rocm_abi_layout_probe.c");
    fs::write(&c_source_path, &c_source).map_err(|error| {
        format!(
            "failed to write C ABI layout probe {}: {error}",
            c_source_path.display()
        )
    })?;

    let mut compiler_version_command = Command::new(compiler);
    compiler_version_command.arg("--version");
    let compiler_version = checked_output(&mut compiler_version_command, "C compiler version")?;
    let compiler_version = String::from_utf8(compiler_version.stdout)
        .map_err(|error| format!("C compiler version output is not UTF-8: {error}"))?;

    let native_probe = output_directory.join("rocm_abi_layout_probe");
    let mut native_compile = Command::new(compiler);
    native_compile.args([
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-DSIM_ROCM_MEASURE=1",
    ]);
    native_compile
        .arg(&c_source_path)
        .arg("-o")
        .arg(&native_probe);
    checked_output(&mut native_compile, "native C ABI layout probe compilation")?;
    let mut native_run = Command::new(&native_probe);
    let c_measurements = parse_measurements(
        &checked_output(&mut native_run, "native C ABI layout probe execution")?.stdout,
        "C",
    )?;

    let mut cross_compile_arguments = vec![
        "-std=c11".to_owned(),
        "-Wall".to_owned(),
        "-Wextra".to_owned(),
        "-Werror".to_owned(),
        "-ffreestanding".to_owned(),
    ];
    if compiler_version.to_ascii_lowercase().contains("clang") {
        cross_compile_arguments.extend(["-target".to_owned(), REQUIRED_C_ABI_TARGET.to_owned()]);
    } else if cfg!(target_os = "linux") {
        cross_compile_arguments.push("-m64".to_owned());
    } else {
        return Err(format!(
            "completion evidence requires a Clang-compatible C compiler to prove the {REQUIRED_C_ABI_TARGET} layout on this host"
        ));
    }
    let cross_object = output_directory.join("rocm_abi_layout_probe.x86_64-linux.o");
    let mut cross_compile = Command::new(compiler);
    cross_compile
        .args(&cross_compile_arguments)
        .arg("-c")
        .arg(&c_source_path)
        .arg("-o")
        .arg(&cross_object);
    checked_output(
        &mut cross_compile,
        "x86_64 Linux C ABI static-assertion compilation",
    )?;
    let cross_object_bytes = fs::read(&cross_object).map_err(|error| {
        format!(
            "failed to read x86_64 Linux C ABI probe object {}: {error}",
            cross_object.display()
        )
    })?;
    if cross_object_bytes.len() < 20
        || &cross_object_bytes[..4] != b"\x7fELF"
        || cross_object_bytes[4] != 2
        || cross_object_bytes[5] != 1
        || cross_object_bytes[18] != 0x3e
        || cross_object_bytes[19] != 0
    {
        return Err(
            "C ABI static-assertion probe did not produce an ELF64 little-endian x86-64 object"
                .to_owned(),
        );
    }

    let rust_source = render_rust_layout_probe(layouts)?;
    let rust_source_path = output_directory.join("rocm_abi_layout_probe.rs");
    fs::write(&rust_source_path, &rust_source).map_err(|error| {
        format!(
            "failed to write Rust ABI layout probe {}: {error}",
            rust_source_path.display()
        )
    })?;
    let mut rustc_version_command = Command::new(rustc);
    rustc_version_command.arg("-vV");
    let rustc_version = checked_output(&mut rustc_version_command, "rustc version")?;
    let rustc_version = String::from_utf8(rustc_version.stdout)
        .map_err(|error| format!("rustc version output is not UTF-8: {error}"))?;
    let rust_probe = output_directory.join("rocm_abi_layout_probe_rust");
    let mut rust_compile = Command::new(rustc);
    rust_compile
        .args(["--edition=2024"])
        .arg(&rust_source_path)
        .arg("-o")
        .arg(&rust_probe);
    checked_output(&mut rust_compile, "Rust ABI layout probe compilation")?;
    let mut rust_run = Command::new(&rust_probe);
    let rust_measurements = parse_measurements(
        &checked_output(&mut rust_run, "Rust ABI layout probe execution")?.stdout,
        "Rust",
    )?;

    let expected: Vec<_> = layouts
        .iter()
        .map(|layout| LayoutMeasurement {
            name: layout.c_name.clone(),
            size: layout.size,
            align: layout.align,
        })
        .collect();
    if c_measurements != expected || rust_measurements != expected {
        return Err(format!(
            "compiled ABI layouts differ: expected {expected:?}, C measured {c_measurements:?}, Rust measured {rust_measurements:?}"
        ));
    }

    Ok(ProbeEvidence {
        c_compiler_version: compiler_version.trim().to_owned(),
        rustc_version: rustc_version.trim().to_owned(),
        c_source_sha256: sha256_hex(c_source.as_bytes()),
        cross_object_sha256: sha256_hex(&cross_object_bytes),
        rust_source_sha256: sha256_hex(rust_source.as_bytes()),
        c_measurements,
        rust_measurements,
        cross_compile_arguments,
    })
}

const REQUIRED_C_ABI_TARGET: &str = "x86_64-unknown-linux-gnu";

fn render_c_layout_probe(layouts: &[Layout], constant_declarations: &[(String, String)]) -> String {
    let mut source = String::from(
        "#include <stddef.h>\n#define __HIP_NODISCARD\n#define SIM_HIP_INIT_FLAGS_ZERO 0u\n#ifdef SIM_ROCM_MEASURE\n#include <stdio.h>\n#endif\n\n",
    );
    for layout in layouts {
        source.push_str(&format!(
            "#line {} {:?}\n{}\n\n",
            layout.line_start, layout.header, layout.declaration
        ));
    }
    for (name, declaration) in constant_declarations {
        if name != "hipInitFlags" {
            source.push_str(&format!(
                "#line 1 \"reviewed-constant-set\"\n{declaration}\n\n"
            ));
        }
    }
    source.push_str("#line 1 \"sim_rocm_layout_assertions\"\n");
    for layout in layouts {
        source.push_str(&format!(
            "_Static_assert(sizeof({}) == {}, \"{} size\");\n_Static_assert(_Alignof({}) == {}, \"{} alignment\");\n",
            layout.c_name,
            layout.size,
            layout.c_name,
            layout.c_name,
            layout.align,
            layout.c_name
        ));
    }
    for (name, _) in constant_declarations {
        if let Some(constants) = constant_specs(name) {
            for constant in constants {
                source.push_str(&format!(
                    "_Static_assert({} == {}, {:?});\n",
                    constant.c_name, constant.value, constant.c_name
                ));
            }
        }
    }
    source.push_str("\n#ifdef SIM_ROCM_MEASURE\nint main(void) {\n");
    for layout in layouts {
        source.push_str(&format!(
            "    printf(\"{}\\t%zu\\t%zu\\n\", sizeof({}), _Alignof({}));\n",
            layout.c_name, layout.c_name, layout.c_name
        ));
    }
    source.push_str("    return 0;\n}\n#endif\n");
    source
}

fn render_rust_layout_probe(layouts: &[Layout]) -> Result<String, String> {
    let mut source =
        String::from("use std::ffi::{c_char, c_int};\nuse std::mem::{align_of, size_of};\n\n");
    for layout in layouts {
        source.push_str(&generate_layout(layout)?);
    }
    source.push_str("fn main() {\n");
    for layout in layouts {
        source.push_str(&format!(
            "    println!(\"{}\\t{{}}\\t{{}}\", size_of::<{}>(), align_of::<{}>());\n",
            layout.c_name, layout.rust_name, layout.rust_name
        ));
    }
    source.push_str("}\n");
    Ok(source)
}

#[allow(
    clippy::disallowed_methods,
    reason = "the opt-in completion audit must synchronously run the configured local C and Rust compilers before emitting evidence"
)]
fn checked_output(command: &mut Command, label: &str) -> Result<Output, String> {
    let command_debug = format!("{command:?}");
    let output = command
        .output()
        .map_err(|error| format!("failed to run {label} ({command_debug}): {error}"))?;
    if output.status.success() {
        return Ok(output);
    }
    Err(format!(
        "{label} failed ({command_debug}) with {}:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    ))
}

fn parse_measurements(bytes: &[u8], label: &str) -> Result<Vec<LayoutMeasurement>, String> {
    let output = std::str::from_utf8(bytes)
        .map_err(|error| format!("{label} layout probe output is not UTF-8: {error}"))?;
    let mut measurements = Vec::new();
    for line in output.lines() {
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 3 {
            return Err(format!("invalid {label} layout measurement line {line:?}"));
        }
        measurements.push(LayoutMeasurement {
            name: fields[0].to_owned(),
            size: parse_usize(fields[1], "measured layout size")?,
            align: parse_usize(fields[2], "measured layout alignment")?,
        });
    }
    Ok(measurements)
}

fn render_completion_artifact(
    headers: &[Header],
    symbols: &[Symbol],
    layouts: &[Layout],
    constant_sets: &[ConstantSet],
    ledger: &str,
    run_id: &str,
    compiler: &OsString,
    rustc: &OsString,
    probe: &ProbeEvidence,
) -> String {
    let mut artifact = String::from("{\n  \"schema_version\": 1,\n");
    artifact.push_str(&format!(
        "  \"status\": \"verified\",\n  \"run_id\": {:?},\n  \"rocm_release\": \"6.1.2\",\n  \"required_c_abi_target\": {:?},\n  \"ledger_sha256\": {:?},\n  \"symbol_count\": {},\n  \"binding_count\": {},\n",
        run_id,
        REQUIRED_C_ABI_TARGET,
        sha256_hex(ledger.as_bytes()),
        symbols.len(),
        symbols.len()
            + layouts.len()
            + constant_sets
                .iter()
                .filter_map(|constant_set| constant_specs(&constant_set.name))
                .map(<[ConstantSpec]>::len)
                .sum::<usize>()
    ));
    artifact.push_str("  \"headers\": [\n");
    for (index, header) in headers.iter().enumerate() {
        artifact.push_str(&format!(
            "    {{\"name\": {:?}, \"source_url\": {:?}, \"byte_length\": {}, \"sha256\": {:?}}}{}\n",
            header.name,
            header.source_url,
            header.byte_length,
            header.digest,
            if index + 1 == headers.len() { "" } else { "," }
        ));
    }
    artifact.push_str("  ],\n  \"layout_declarations\": [\n");
    for (index, layout) in layouts.iter().enumerate() {
        artifact.push_str(&format!(
            "    {{\"name\": {:?}, \"header\": {:?}, \"line_start\": {}, \"line_end\": {}, \"excerpt_sha256\": {:?}}}{}\n",
            layout.c_name,
            layout.header,
            layout.line_start,
            layout.line_end,
            layout.digest,
            if index + 1 == layouts.len() { "" } else { "," }
        ));
    }
    artifact.push_str("  ],\n  \"constant_declarations\": [\n");
    let constant_count = constant_sets
        .iter()
        .filter_map(|constant_set| constant_specs(&constant_set.name))
        .map(<[ConstantSpec]>::len)
        .sum::<usize>();
    let mut emitted_constants = 0;
    for constant_set in constant_sets {
        for constant in constant_specs(&constant_set.name).unwrap_or_default() {
            emitted_constants += 1;
            artifact.push_str(&format!(
                "    {{\"name\": {:?}, \"header\": {:?}, \"line_start\": {}, \"line_end\": {}, \"excerpt_sha256\": {:?}}}{}\n",
                constant.c_name,
                constant_set.header,
                constant_set.line_start,
                constant_set.line_end,
                constant_set.digest,
                if emitted_constants == constant_count { "" } else { "," }
            ));
        }
    }
    artifact.push_str("  ],\n  \"c_probe\": {\n");
    artifact.push_str(&format!(
        "    \"compiler\": {:?},\n    \"compiler_version\": {:?},\n    \"source_sha256\": {:?},\n    \"cross_compile_arguments\": [{}],\n    \"cross_object_format\": \"ELF64 little-endian x86-64\",\n    \"cross_object_sha256\": {:?},\n    \"static_assertions\": \"passed\",\n    \"measurements\": {}\n",
        compiler.to_string_lossy(),
        probe.c_compiler_version,
        probe.c_source_sha256,
        probe
            .cross_compile_arguments
            .iter()
            .map(|argument| format!("{argument:?}"))
            .collect::<Vec<_>>()
            .join(", "),
        probe.cross_object_sha256,
        measurements_json(&probe.c_measurements)
    ));
    artifact.push_str("  },\n  \"rust_probe\": {\n");
    artifact.push_str(&format!(
        "    \"rustc\": {:?},\n    \"rustc_version\": {:?},\n    \"source_sha256\": {:?},\n    \"measurements\": {}\n",
        rustc.to_string_lossy(),
        probe.rustc_version,
        probe.rust_source_sha256,
        measurements_json(&probe.rust_measurements)
    ));
    artifact.push_str(
        "  },\n  \"completion_command\": \"bash crates/sim_backend_rocm/abi/verify-completion-evidence.sh <flat-pinned-header-directory> [artifact-path]\"\n}\n",
    );
    artifact
}

fn measurements_json(measurements: &[LayoutMeasurement]) -> String {
    format!(
        "[{}]",
        measurements
            .iter()
            .map(|measurement| format!(
                "{{\"name\": {:?}, \"size\": {}, \"align\": {}}}",
                measurement.name, measurement.size, measurement.align
            ))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn parse_evidence(
    input: &str,
) -> Result<(Vec<Header>, Vec<Symbol>, Vec<Layout>, Vec<ConstantSet>), String> {
    let mut lines = input.lines().peekable();
    if lines.next() != Some("schema\t1") {
        return Err("reviewed ABI evidence has an unsupported schema".to_owned());
    }

    let mut headers = Vec::new();
    let mut symbols = Vec::new();
    let mut layouts = Vec::new();
    let mut constant_sets = Vec::new();
    while let Some(line) = lines.next() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with("@@HEADER\t") {
            let fields = fields(line, 6)?;
            if fields[5] != "MIT" {
                return Err(format!("unsupported header license for {}", fields[1]));
            }
            headers.push(Header {
                name: fields[1].to_owned(),
                source_url: fields[2].to_owned(),
                byte_length: parse_usize(fields[3], "header byte length")?,
                digest: fields[4].to_owned(),
            });
            continue;
        }
        if line.starts_with("@@SYMBOL\t") {
            let fields = fields(line, 8)?;
            let declaration = read_declaration(&mut lines)?;
            symbols.push(Symbol {
                library: fields[1].to_owned(),
                name: fields[2].to_owned(),
                alias: fields[3].to_owned(),
                header: fields[4].to_owned(),
                line_start: parse_usize(fields[5], "symbol start line")?,
                line_end: parse_usize(fields[6], "symbol end line")?,
                digest: fields[7].to_owned(),
                declaration,
            });
            continue;
        }
        if line.starts_with("@@LAYOUT\t") {
            let fields = fields(line, 9)?;
            let declaration = read_declaration(&mut lines)?;
            layouts.push(Layout {
                c_name: fields[1].to_owned(),
                rust_name: fields[2].to_owned(),
                header: fields[3].to_owned(),
                line_start: parse_usize(fields[4], "layout start line")?,
                line_end: parse_usize(fields[5], "layout end line")?,
                digest: fields[6].to_owned(),
                size: parse_usize(fields[7], "layout size")?,
                align: parse_usize(fields[8], "layout alignment")?,
                declaration,
            });
            continue;
        }
        if line.starts_with("@@CONSTANT_SET\t") {
            let fields = fields(line, 7)?;
            constant_sets.push(ConstantSet {
                name: fields[1].to_owned(),
                header: fields[2].to_owned(),
                line_start: parse_usize(fields[3], "constant-set start line")?,
                line_end: parse_usize(fields[4], "constant-set end line")?,
                digest: fields[5].to_owned(),
            });
            let expected_values = constant_specs(fields[1])
                .ok_or_else(|| format!("unreviewed ABI constant set {}", fields[1]))?
                .iter()
                .map(|constant| constant.value.to_string())
                .collect::<Vec<_>>()
                .join(",");
            if fields[6] != expected_values {
                return Err(format!(
                    "reviewed ABI constant set {} values differ from {expected_values}",
                    fields[1]
                ));
            }
            continue;
        }
        return Err(format!("unrecognized reviewed ABI evidence line: {line}"));
    }

    if headers.len() != 9 || symbols.len() != 52 || layouts.len() != 3 || constant_sets.len() != 9 {
        return Err(format!(
            "reviewed ABI evidence must contain 9 headers, 52 symbols, 3 layouts, and 9 constant sets; found {}, {}, {}, and {}",
            headers.len(),
            symbols.len(),
            layouts.len(),
            constant_sets.len()
        ));
    }
    Ok((headers, symbols, layouts, constant_sets))
}

fn fields(line: &str, expected: usize) -> Result<Vec<&str>, String> {
    let fields: Vec<_> = line.split('\t').collect();
    if fields.len() != expected {
        return Err(format!(
            "reviewed ABI evidence row has {} fields instead of {expected}: {line}",
            fields.len()
        ));
    }
    Ok(fields)
}

fn parse_usize(value: &str, label: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|error| format!("invalid {label} {value:?}: {error}"))
}

fn read_declaration<'a>(
    lines: &mut std::iter::Peekable<impl Iterator<Item = &'a str>>,
) -> Result<String, String> {
    let mut declaration = Vec::new();
    loop {
        let line = lines
            .next()
            .ok_or_else(|| "unterminated reviewed ABI declaration".to_owned())?;
        if line == "@@END" {
            break;
        }
        declaration.push(line);
    }
    if declaration.is_empty() {
        return Err("reviewed ABI declaration cannot be empty".to_owned());
    }
    Ok(declaration.join("\n"))
}

fn validate_headers(headers: &[Header], manifest: &str, catalog: &str) -> Result<(), String> {
    for (expected_name, expected_url, expected_length, expected_digest) in EXPECTED_HEADERS {
        let header = headers
            .iter()
            .find(|header| header.name == expected_name)
            .ok_or_else(|| format!("missing reviewed header record {expected_name}"))?;
        if header.source_url != expected_url
            || header.byte_length != expected_length
            || header.digest != expected_digest
        {
            return Err(format!(
                "reviewed header record {expected_name} differs from the verified ROCm 6.1.2 bytes"
            ));
        }
        if manifest.matches(expected_digest).count() != 1 {
            return Err(format!(
                "runtime manifest must contain the verified {expected_name} SHA-256 exactly once"
            ));
        }
    }
    let manifest_reference = "\"symbol_manifest\": \"crates/sim_backend_rocm/abi/symbols-v1.json\"";
    if catalog.matches(manifest_reference).count() != 1 {
        return Err("spec catalog must point to the checked runtime symbol manifest".to_owned());
    }
    Ok(())
}

fn validate_unique_records(
    symbols: &[Symbol],
    layouts: &[Layout],
    constant_sets: &[ConstantSet],
) -> Result<(), String> {
    let mut symbol_names = BTreeSet::new();
    let mut aliases = BTreeSet::new();
    for symbol in symbols {
        if !symbol_names.insert((&symbol.library, &symbol.name)) {
            return Err(format!(
                "duplicate reviewed ABI symbol {}:{}",
                symbol.library, symbol.name
            ));
        }
        if !aliases.insert(&symbol.alias) {
            return Err(format!(
                "duplicate reviewed Rust ABI alias {}",
                symbol.alias
            ));
        }
    }
    let mut layout_names = BTreeSet::new();
    for layout in layouts {
        if !layout_names.insert(&layout.c_name) {
            return Err(format!("duplicate reviewed ABI layout {}", layout.c_name));
        }
    }
    let mut constant_names = BTreeSet::new();
    for constant_set in constant_sets {
        if !constant_names.insert(constant_set.name.as_str())
            || constant_set.digest.len() != 64
            || constant_specs(&constant_set.name).is_none()
        {
            return Err(format!(
                "reviewed ABI constant set {} is invalid or duplicated",
                constant_set.name
            ));
        }
    }
    Ok(())
}

fn generate_bindings(
    headers: &[Header],
    symbols: &[Symbol],
    layouts: &[Layout],
    constant_sets: &[ConstantSet],
    manifest: &str,
    completion: Option<&CompletionEvidence>,
) -> Result<String, String> {
    let header_names: BTreeSet<_> = headers.iter().map(|header| header.name.as_str()).collect();
    let mut generated =
        String::from("// @generated by build.rs from abi/reviewed-bindings-v1.txt\n");

    for constant_set in constant_sets {
        if !header_names.contains(constant_set.header.as_str()) {
            return Err(format!(
                "constant set {} references unknown header {}",
                constant_set.name, constant_set.header
            ));
        }
    }
    for constant_set in constant_sets {
        for constant in constant_specs(&constant_set.name).unwrap_or_default() {
            generated.push_str(&format!(
                "pub(crate) const {}: {} = {};\n",
                constant.rust_name, constant.rust_type, constant.value
            ));
        }
    }
    generated.push('\n');

    for symbol in symbols {
        validate_excerpt(
            "symbol",
            &symbol.name,
            &symbol.header,
            symbol.line_start,
            symbol.line_end,
            &symbol.digest,
            &symbol.declaration,
            &header_names,
        )?;
        let (canonical_signature, rust_arguments, rust_return) =
            parse_function_declaration(&symbol.name, &symbol.declaration)?;
        validate_contract_document(
            "runtime manifest",
            manifest,
            &symbol.name,
            &canonical_signature,
        )?;
        generated.push_str(&format!(
            "pub(crate) type {} = unsafe extern \"C\" fn({}) -> {};\n",
            symbol.alias, rust_arguments, rust_return
        ));
    }

    generated.push_str("\npub(crate) const SYMBOLS: &[SymbolContract] = &[\n");
    for symbol in symbols {
        let (canonical_signature, _, _) =
            parse_function_declaration(&symbol.name, &symbol.declaration)?;
        generated.push_str(&format!(
            "    SymbolContract {{ library: {:?}, name: {:?}, signature: {:?} }},\n",
            symbol.library, symbol.name, canonical_signature
        ));
    }
    generated.push_str("] ;\n\n");

    generated.push_str("pub(crate) const LIBRARIES: &[(&str, &str)] = &[\n");
    for (library, file) in [
        ("libamdhip64", "libamdhip64.so"),
        ("libhiprtc", "libhiprtc.so"),
        ("librocblas", "librocblas.so"),
        ("libMIOpen", "libMIOpen.so"),
    ] {
        generated.push_str(&format!("    ({library:?}, {file:?}),\n"));
    }
    generated.push_str("];\n\n");

    generated.push_str("pub(crate) const HEADER_EVIDENCE: &[HeaderEvidence] = &[\n");
    for header in headers {
        generated.push_str(&format!(
            "    HeaderEvidence {{ name: {:?}, source_url: {:?}, byte_length: {}, sha256: {:?} }},\n",
            header.name, header.source_url, header.byte_length, header.digest
        ));
    }
    generated.push_str("];\n\n");

    generated.push_str("pub(crate) const BINDING_EVIDENCE: &[BindingEvidence] = &[\n");
    for symbol in symbols {
        generated.push_str(&format!(
            "    BindingEvidence {{ name: {:?}, header: {:?}, line_start: {}, line_end: {}, excerpt_sha256: {:?} }},\n",
            symbol.name, symbol.header, symbol.line_start, symbol.line_end, symbol.digest
        ));
    }
    for layout in layouts {
        generated.push_str(&format!(
            "    BindingEvidence {{ name: {:?}, header: {:?}, line_start: {}, line_end: {}, excerpt_sha256: {:?} }},\n",
            layout.c_name, layout.header, layout.line_start, layout.line_end, layout.digest
        ));
    }
    for constant_set in constant_sets {
        for constant in constant_specs(&constant_set.name).unwrap_or_default() {
            generated.push_str(&format!(
                "    BindingEvidence {{ name: {:?}, header: {:?}, line_start: {}, line_end: {}, excerpt_sha256: {:?} }},\n",
                constant.c_name,
                constant_set.header,
                constant_set.line_start,
                constant_set.line_end,
                constant_set.digest
            ));
        }
    }
    generated.push_str("];\n\n");

    match completion {
        Some(completion) => generated.push_str(&format!(
            "pub(crate) const COMPLETION_EVIDENCE: CompletionEvidence = CompletionEvidence::Verified {{ artifact_sha256: {:?}, run_id: {:?} }};\n\n",
            completion.artifact_sha256, completion.run_id
        )),
        None => generated.push_str(
            "pub(crate) const COMPLETION_EVIDENCE: CompletionEvidence = CompletionEvidence::Unavailable;\n\n",
        ),
    }

    for layout in layouts {
        validate_excerpt(
            "layout",
            &layout.c_name,
            &layout.header,
            layout.line_start,
            layout.line_end,
            &layout.digest,
            &layout.declaration,
            &header_names,
        )?;
        let manifest_entry = format!(
            "{{ \"name\": {:?}, \"size\": {}, \"align\": {} }}",
            layout.c_name, layout.size, layout.align
        );
        if !manifest.contains(&manifest_entry) {
            return Err(format!(
                "layout {} is not identical in the runtime manifest",
                layout.c_name
            ));
        }
        generated.push_str(&generate_layout(layout)?);
    }
    Ok(generated)
}

fn validate_excerpt(
    kind: &str,
    name: &str,
    header: &str,
    line_start: usize,
    line_end: usize,
    expected_digest: &str,
    declaration: &str,
    header_names: &BTreeSet<&str>,
) -> Result<(), String> {
    if !header_names.contains(header) {
        return Err(format!("{kind} {name} references unknown header {header}"));
    }
    let actual_lines = declaration.lines().count();
    if line_end < line_start || line_end - line_start + 1 != actual_lines {
        return Err(format!(
            "{kind} {name} has inconsistent upstream line evidence"
        ));
    }
    let actual_digest = sha256_hex(declaration.as_bytes());
    if actual_digest != expected_digest {
        return Err(format!(
            "{kind} {name} declaration bytes differ from reviewed header excerpt: expected {expected_digest}, found {actual_digest}"
        ));
    }
    Ok(())
}

fn validate_contract_document(
    label: &str,
    document: &str,
    name: &str,
    signature: &str,
) -> Result<(), String> {
    let entry = format!("{{ \"name\": {name:?}, \"signature\": {signature:?} }}");
    if document.matches(&entry).count() != 1 {
        return Err(format!(
            "{label} must contain exactly one C-derived contract {name}: {signature}"
        ));
    }
    Ok(())
}

fn parse_function_declaration(
    name: &str,
    declaration: &str,
) -> Result<(String, String, String), String> {
    let normalized = declaration.replace("__dparm(0)", "").replace(" = NULL", "");
    let normalized = collapse_whitespace(&normalized);
    let normalized = normalized
        .strip_suffix(';')
        .ok_or_else(|| format!("reviewed declaration for {name} lacks a semicolon"))?;
    let name_offset = normalized
        .find(name)
        .ok_or_else(|| format!("reviewed declaration does not declare {name}"))?;
    let return_type = normalized[..name_offset]
        .trim()
        .strip_prefix("ROCBLAS_EXPORT ")
        .or_else(|| {
            normalized[..name_offset]
                .trim()
                .strip_prefix("MIOPEN_EXPORT ")
        })
        .unwrap_or(normalized[..name_offset].trim());
    let after_name = &normalized[name_offset + name.len()..];
    let parameters = after_name
        .strip_prefix('(')
        .and_then(|parameters| parameters.strip_suffix(')'))
        .ok_or_else(|| format!("reviewed declaration for {name} has invalid parentheses"))?;

    let return_type = canonical_c_type(return_type);
    let mut canonical_parameters = Vec::new();
    if parameters.trim() != "void" && !parameters.trim().is_empty() {
        for parameter in parameters.split(',') {
            canonical_parameters.push(parameter_type(parameter)?);
        }
    }
    let canonical_parameter_text = if canonical_parameters.is_empty() {
        "void".to_owned()
    } else {
        canonical_parameters.join(",")
    };
    let canonical_signature = format!("{return_type}({canonical_parameter_text})");
    let rust_arguments = canonical_parameters
        .iter()
        .map(|parameter| rust_type(parameter))
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    let rust_return = rust_type(&return_type)?;
    Ok((canonical_signature, rust_arguments, rust_return.to_owned()))
}

fn parameter_type(parameter: &str) -> Result<String, String> {
    let parameter = parameter.trim();
    let (type_name, identifier) = parameter
        .rsplit_once(char::is_whitespace)
        .ok_or_else(|| format!("parameter lacks a name: {parameter}"))?;
    if identifier.contains('*') || identifier.is_empty() {
        return Err(format!("invalid parameter identifier in {parameter}"));
    }
    Ok(canonical_c_type(type_name))
}

fn canonical_c_type(type_name: &str) -> String {
    let mut type_name = collapse_whitespace(type_name);
    while type_name.contains(" *") || type_name.contains("* ") {
        type_name = type_name.replace(" *", "*").replace("* ", "*");
    }
    match type_name.as_str() {
        "unsigned" => "unsigned int".to_owned(),
        "const char**" => "const char* const*".to_owned(),
        "const miopenTensorDescriptor_t" => "miopenTensorDescriptor_t".to_owned(),
        "const miopenConvolutionDescriptor_t" => "miopenConvolutionDescriptor_t".to_owned(),
        _ => type_name,
    }
}

fn collapse_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn rust_type(c_type: &str) -> Result<&'static str, String> {
    let rust_type = match c_type {
        "hipError_t" => "HipError",
        "hiprtcResult" => "HipRtcResult",
        "rocblas_status" => "RocblasStatus",
        "miopenStatus_t" => "MiopenStatus",
        "unsigned int" => "c_uint",
        "int"
        | "hipDevice_t"
        | "rocblas_int"
        | "miopenDataType_t"
        | "miopenConvolutionMode_t"
        | "miopenConvFwdAlgorithm_t" => "c_int",
        "hipDeviceAttribute_t" => "HipDeviceAttribute",
        "hipMemcpyKind" => "HipMemcpyKind",
        "rocblas_operation" => "RocblasOperation",
        "size_t" => "usize",
        "int*" => "*mut c_int",
        "size_t*" => "*mut usize",
        "void*" => "*mut c_void",
        "void**" => "*mut *mut c_void",
        "const void*" => "*const c_void",
        "char*" => "*mut c_char",
        "const char*" => "*const c_char",
        "const char* const*" => "*const *const c_char",
        "float*" => "*mut f32",
        "const float*" => "*const f32",
        "hipStream_t" | "miopenAcceleratorQueue_t" => "HipStream",
        "hipStream_t*" => "*mut HipStream",
        "hipEvent_t" => "HipEvent",
        "hipEvent_t*" => "*mut HipEvent",
        "hipModule_t" => "HipModule",
        "hipModule_t*" => "*mut HipModule",
        "hipFunction_t" => "HipFunction",
        "hipFunction_t*" => "*mut HipFunction",
        "hiprtcProgram" => "HipRtcProgram",
        "hiprtcProgram*" => "*mut HipRtcProgram",
        "rocblas_handle" => "RocblasHandle",
        "rocblas_handle*" => "*mut RocblasHandle",
        "miopenHandle_t" => "MiopenHandle",
        "miopenHandle_t*" => "*mut MiopenHandle",
        "miopenTensorDescriptor_t" => "MiopenTensorDescriptor",
        "miopenTensorDescriptor_t*" => "*mut MiopenTensorDescriptor",
        "miopenConvolutionDescriptor_t" => "MiopenConvolutionDescriptor",
        "miopenConvolutionDescriptor_t*" => "*mut MiopenConvolutionDescriptor",
        _ => return Err(format!("unreviewed ROCm C ABI type {c_type}")),
    };
    Ok(rust_type)
}

fn generate_layout(layout: &Layout) -> Result<String, String> {
    let code = match layout.c_name.as_str() {
        "hipUUID"
            if layout.rust_name == "HipUuid"
                && layout.size == 16
                && layout.align == 1
                && layout.declaration.contains("typedef struct hipUUID_t {")
                && layout.declaration.contains("char bytes[16];") =>
        {
            "#[repr(C)]\npub(crate) struct HipUuid {\n    pub(crate) bytes: [c_char; 16],\n}\n\n"
        }
        "hipIpcMemHandle_t"
            if layout.rust_name == "HipIpcMemHandle"
                && layout.size == 64
                && layout.align == 1
                && layout
                    .declaration
                    .contains("#define HIP_IPC_HANDLE_SIZE 64")
                && layout
                    .declaration
                    .contains("char reserved[HIP_IPC_HANDLE_SIZE];") =>
        {
            "#[repr(C)]\npub(crate) struct HipIpcMemHandle {\n    pub(crate) reserved: [c_char; 64],\n}\n\n"
        }
        "miopenConvAlgoPerf_t"
            if layout.rust_name == "MiopenConvAlgoPerf"
                && layout.size == 16
                && layout.align == 8
                && layout.declaration.contains("} miopenConvFwdAlgorithm_t;")
                && layout
                    .declaration
                    .contains("} miopenConvBwdWeightsAlgorithm_t;")
                && layout
                    .declaration
                    .contains("} miopenConvBwdDataAlgorithm_t;")
                && layout.declaration.contains("float time;")
                && layout.declaration.contains("size_t memory;") =>
        {
            "#[repr(C)]\npub(crate) struct MiopenConvAlgoPerf {\n    pub(crate) algorithm: c_int,\n    pub(crate) time: f32,\n    pub(crate) memory: usize,\n}\n\n"
        }
        _ => {
            return Err(format!(
                "layout {} is not derivable from its reviewed ROCm declaration bytes",
                layout.c_name
            ));
        }
    };
    Ok(code.to_owned())
}

fn sha256_hex(input: &[u8]) -> String {
    const INITIAL: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const ROUND: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let bit_length = (input.len() as u64).wrapping_mul(8);
    let mut message = input.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_length.to_be_bytes());

    let mut hash = INITIAL;
    for chunk in message.chunks_exact(64) {
        let mut schedule = [0_u32; 64];
        for (index, bytes) in chunk.chunks_exact(4).enumerate() {
            schedule[index] = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        for index in 16..64 {
            let small_zero = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let small_one = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(small_zero)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(small_one);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = hash;
        for index in 0..64 {
            let big_one = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temporary_one = h
                .wrapping_add(big_one)
                .wrapping_add(choose)
                .wrapping_add(ROUND[index])
                .wrapping_add(schedule[index]);
            let big_zero = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temporary_two = big_zero.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary_one);
            d = c;
            c = b;
            b = a;
            a = temporary_one.wrapping_add(temporary_two);
        }
        hash[0] = hash[0].wrapping_add(a);
        hash[1] = hash[1].wrapping_add(b);
        hash[2] = hash[2].wrapping_add(c);
        hash[3] = hash[3].wrapping_add(d);
        hash[4] = hash[4].wrapping_add(e);
        hash[5] = hash[5].wrapping_add(f);
        hash[6] = hash[6].wrapping_add(g);
        hash[7] = hash[7].wrapping_add(h);
    }
    hash.iter().map(|word| format!("{word:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEDGER: &str = include_str!("abi/reviewed-bindings-v1.txt");

    #[test]
    fn edited_ledger_declaration_cannot_validate() {
        let (headers, mut symbols, _, _) = parse_evidence(LEDGER).expect("checked ledger parses");
        symbols[0].declaration.push(' ');
        let header_names: BTreeSet<_> = headers.iter().map(|header| header.name.as_str()).collect();
        let symbol = &symbols[0];
        let result = validate_excerpt(
            "symbol",
            &symbol.name,
            &symbol.header,
            symbol.line_start,
            symbol.line_end,
            &symbol.digest,
            &symbol.declaration,
            &header_names,
        );
        assert!(result.is_err());
    }

    #[test]
    fn edited_ledger_with_recomputed_excerpt_hash_still_fails_full_header_proof() {
        let symbol = Symbol {
            library: "fixture".to_owned(),
            name: "altered".to_owned(),
            alias: "Altered".to_owned(),
            header: "fixture.h".to_owned(),
            line_start: 1,
            line_end: 1,
            digest: sha256_hex(b"int altered(void);"),
            declaration: "int altered(void);".to_owned(),
        };
        let result = validate_header_ranges(
            "fixture.h",
            Path::new("fixture.h"),
            "int exact(void);",
            &[symbol],
            &[],
            &[],
        );
        assert!(result.is_err());
    }

    #[test]
    fn missing_full_headers_cannot_validate_completion_evidence() {
        let (headers, symbols, layouts, constant_sets) =
            parse_evidence(LEDGER).expect("checked ledger parses");
        let result = validate_full_header_bytes(
            Path::new("/definitely-missing-sim-rocm-reviewed-headers"),
            &headers,
            &symbols,
            &layouts,
            &constant_sets,
        );
        assert!(result.is_err());
    }

    #[test]
    fn missing_compiled_layout_cannot_validate_completion_evidence() {
        let (_, _, mut layouts, _) = parse_evidence(LEDGER).expect("checked ledger parses");
        layouts.pop();
        assert!(validate_completion_layout_set(&layouts).is_err());
    }
}
