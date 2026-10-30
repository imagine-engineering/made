use std::{
    collections::BTreeSet,
    env, fs, io,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelFamilyBuildEntry {
    pub module: String,
    pub identifier: String,
    pub feature_id: String,
    pub fixture: String,
    pub source_ordinal: u16,
}

fn main() -> io::Result<()> {
    let mut modules = Vec::new();
    let mut names = BTreeSet::new();
    let mut latent_formats = Vec::new();
    let mut model_families = Vec::<ModelFamilyBuildEntry>::new();
    for kind in ["families", "latent_formats", "slices"] {
        let directory = PathBuf::from("src").join(kind);
        println!("cargo:rerun-if-changed={}", directory.display());
        if directory.is_dir() {
            for entry in fs::read_dir(&directory)? {
                let path = entry?.path();
                if is_apple_double_metadata(&path)
                    || path.extension().and_then(|value| value.to_str()) != Some("rs")
                {
                    continue;
                }
                let name = module_name(&path)?;
                if !names.insert(name.to_owned()) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("duplicate model module name: {name}"),
                    ));
                }
                if kind == "latent_formats" {
                    let source = fs::read_to_string(&path)?;
                    let identifier = latent_format_identifier(&source, &path)?;
                    register_latent_format(&mut latent_formats, name, identifier)?;
                } else if kind == "families" {
                    let source = fs::read_to_string(&path)?;
                    let entry = model_family_build_entry(&source, &path, name)?;
                    register_model_family_entry(&mut model_families, entry)?;
                }
                modules.push((kind.to_owned(), name.to_owned()));
            }
        }
    }
    modules.sort();
    latent_formats.sort();
    sort_model_family_entries(&mut model_families);
    let latent_format_test_names = latent_format_test_names(&latent_formats)?;
    let model_family_tuples = model_families
        .iter()
        .map(|entry| {
            (
                entry.module.clone(),
                entry.identifier.clone(),
                entry.feature_id.clone(),
                entry.fixture.clone(),
            )
        })
        .collect::<Vec<_>>();
    model_family_fixture_names(&model_family_tuples)?;
    let model_family_test_names = model_family_test_names(&model_families)?;
    let values = modules
        .iter()
        .map(|(kind, name)| format!("\"{kind}/{name}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let includes = modules
        .iter()
        .map(|(kind, name)| {
            format!(
                "pub mod generated_{name} {{ include!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/src/{kind}/{name}.rs\")); }}\n"
            )
        })
        .collect::<String>();
    let latent_registry = latent_formats
        .iter()
        .map(|(name, _)| format!("generated_{name}::LATENT_FORMAT"))
        .collect::<Vec<_>>()
        .join(", ");
    let latent_manifest = latent_formats
        .iter()
        .map(|(name, _)| format!("(\"{name}\", &generated_{name}::LATENT_FORMAT)"))
        .collect::<Vec<_>>()
        .join(", ");
    let model_family_registry = model_families
        .iter()
        .map(|entry| format!("generated_{}::MODEL_FAMILY", entry.module))
        .collect::<Vec<_>>()
        .join(", ");
    let model_family_registration_registry = model_families
        .iter()
        .map(|entry| format!("generated_{}::MODEL_FAMILY_REGISTRATION", entry.module))
        .collect::<Vec<_>>()
        .join(", ");
    let model_family_manifest = model_families
        .iter()
        .map(|entry| {
            format!(
                "(\"{}\", &generated_{}::MODEL_FAMILY)",
                entry.fixture, entry.module
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let model_family_source_manifest = model_families
        .iter()
        .map(|entry| {
            format!(
                "(\"{}\", \"{}\", \"{}\", {})",
                entry.module, entry.feature_id, entry.fixture, entry.source_ordinal
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let model_family_module_values = model_families
        .iter()
        .map(|entry| format!("\"{}\"", entry.module))
        .collect::<Vec<_>>()
        .join(", ");
    let model_family_feature_values = model_families
        .iter()
        .map(|entry| format!("\"{}\"", entry.feature_id))
        .collect::<Vec<_>>()
        .join(", ");
    let model_family_identifier_values = model_families
        .iter()
        .map(|entry| format!("\"{}\"", entry.identifier))
        .collect::<Vec<_>>()
        .join(", ");
    let model_family_fixture_values = model_families
        .iter()
        .map(|entry| format!("\"{}\"", entry.fixture))
        .collect::<Vec<_>>()
        .join(", ");
    let model_family_source_ordinal_values = model_families
        .iter()
        .map(|entry| entry.source_ordinal.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let output_directory = PathBuf::from(env::var_os("OUT_DIR").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "Cargo did not provide the OUT_DIR build-script variable",
        )
    })?);
    fs::write(
        output_directory.join("generated_modules.rs"),
        format!(
            "{includes}pub const GENERATED_MODULES: &[&str] = &[{values}];\n\
             pub const GENERATED_LATENT_FORMATS: &[crate::LatentFormatDefinition] = &[{latent_registry}];\n\
             pub const GENERATED_LATENT_FORMAT_MANIFEST: &[(&str, &crate::LatentFormatDefinition)] = &[{latent_manifest}];\n\
             pub const GENERATED_MODEL_FAMILIES: &[crate::ModelFamilyDefinition] = &[{model_family_registry}];\n\
             pub const GENERATED_MODEL_FAMILY_MANIFEST: &[(&str, &crate::ModelFamilyDefinition)] = &[{model_family_manifest}];\n\
             pub const GENERATED_MODEL_FAMILY_REGISTRATIONS: &[crate::ModelFamilyRegistration] = &[{model_family_registration_registry}];\n\
             pub const GENERATED_MODEL_FAMILY_SOURCE_MANIFEST: &[(&str, &str, &str, u16)] = &[{model_family_source_manifest}];\n\
             pub const GENERATED_MODEL_FAMILY_MODULES: &[&str] = &[{model_family_module_values}];\n\
             pub const GENERATED_MODEL_FAMILY_FEATURE_IDS: &[&str] = &[{model_family_feature_values}];\n\
             pub const GENERATED_MODEL_FAMILY_IDENTIFIERS: &[&str] = &[{model_family_identifier_values}];\n\
             pub const GENERATED_MODEL_FAMILY_FIXTURES: &[&str] = &[{model_family_fixture_values}];\n\
             pub const GENERATED_MODEL_FAMILY_SOURCE_ORDINALS: &[u16] = &[{model_family_source_ordinal_values}];\n"
        ),
    )?;
    let test_values = latent_format_test_names
        .iter()
        .map(|name| format!("\"{name}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let test_includes = latent_format_test_names
        .iter()
        .map(|name| {
            format!(
                "mod generated_{name} {{ include!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/tests/latent_formats/{name}.rs\")); }}\n"
            )
        })
        .collect::<String>();
    fs::write(
        output_directory.join("generated_latent_format_tests.rs"),
        format!(
            "{test_includes}const GENERATED_LATENT_FORMAT_TESTS: &[&str] = &[{test_values}];\n\
             #[test]\n\
             fn generated_latent_format_test_manifest_is_sorted_and_unique() {{\n\
                 assert!(GENERATED_LATENT_FORMAT_TESTS.windows(2).all(|pair| pair[0] < pair[1]));\n\
             }}\n"
        ),
    )?;
    fs::write(
        output_directory.join("generated_model_family_tests.rs"),
        format!(
            "{}const GENERATED_MODEL_FAMILY_TEST_MODULES: &[&str] = &[{}];\n\
             const GENERATED_MODEL_FAMILY_TEST_FIXTURES: &[&str] = &[{model_family_fixture_values}];\n\
             #[test]\n\
             fn generated_model_family_test_manifest_matches_source_manifest() {{\n\
                 assert_eq!(GENERATED_MODEL_FAMILY_TEST_MODULES, sim_model::GENERATED_MODEL_FAMILY_MODULES);\n\
                 assert_eq!(GENERATED_MODEL_FAMILY_TEST_FIXTURES, sim_model::GENERATED_MODEL_FAMILY_FIXTURES);\n\
                 assert_eq!(GENERATED_MODEL_FAMILY_TEST_MODULES.len(), sim_model::GENERATED_MODEL_FAMILY_REGISTRATIONS.len());\n\
                 for (index, registration) in sim_model::GENERATED_MODEL_FAMILY_REGISTRATIONS.iter().enumerate() {{\n\
                     let (module, feature_id, fixture, source_ordinal) = sim_model::GENERATED_MODEL_FAMILY_SOURCE_MANIFEST[index];\n\
                     assert_eq!(module, GENERATED_MODEL_FAMILY_TEST_MODULES[index]);\n\
                     assert_eq!(fixture, GENERATED_MODEL_FAMILY_TEST_FIXTURES[index]);\n\
                     assert_eq!(feature_id, registration.definition.feature_id);\n\
                     assert_eq!(sim_model::GENERATED_MODEL_FAMILY_IDENTIFIERS[index], registration.definition.identifier);\n\
                     assert_eq!(source_ordinal, registration.source_ordinal);\n\
                     assert_eq!(source_ordinal, sim_model::GENERATED_MODEL_FAMILY_SOURCE_ORDINALS[index]);\n\
                 }}\n\
             }}\n",
            model_family_test_names
                .iter()
                .map(|name| format!(
                    "mod generated_{name} {{ include!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/tests/families/{name}.rs\")); }}\n"
                ))
                .collect::<String>(),
            model_family_test_names
                .iter()
                .map(|name| format!("\"{name}\""))
                .collect::<Vec<_>>()
                .join(", "),
        ),
    )
}

pub fn model_family_build_entry(
    source: &str,
    path: &std::path::Path,
    module: &str,
) -> io::Result<ModelFamilyBuildEntry> {
    if !valid_module_name(module) {
        return Err(invalid_data(format!(
            "model family module name is invalid: {module}"
        )));
    }
    let identifier = source_constant(source, path, "MODEL_FAMILY_IDENTIFIER")?;
    let feature_id = source_constant(source, path, "MODEL_FAMILY_FEATURE_ID")?;
    let fixture = source_constant(source, path, "MODEL_FAMILY_FIXTURE")?;
    validate_model_family_identity(identifier, feature_id, fixture)?;
    if !source.contains("pub const MODEL_FAMILY:") {
        return Err(invalid_data(format!(
            "model family module must declare MODEL_FAMILY: {}",
            path.display()
        )));
    }
    let registration = source
        .split_once("pub const MODEL_FAMILY_REGISTRATION:")
        .map(|(_, registration)| registration)
        .and_then(|source| source.split_once(";").map(|(value, _)| value))
        .ok_or_else(|| {
            invalid_data(format!(
                "model family module must declare MODEL_FAMILY_REGISTRATION: {}",
                path.display()
            ))
        })?;
    let compact_registration = registration
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>();
    if !compact_registration.contains("ModelFamilyRegistration{")
        || !compact_registration.contains("definition:&MODEL_FAMILY")
    {
        return Err(invalid_data(format!(
            "model family registration must be an immutable ModelFamilyRegistration referencing MODEL_FAMILY: {}",
            path.display()
        )));
    }
    let source_ordinal = source_u16_field(registration, path, "source_ordinal")?;
    Ok(ModelFamilyBuildEntry {
        module: module.to_owned(),
        identifier: identifier.to_owned(),
        feature_id: feature_id.to_owned(),
        fixture: fixture.to_owned(),
        source_ordinal,
    })
}

pub fn register_model_family_entry(
    model_families: &mut Vec<ModelFamilyBuildEntry>,
    entry: ModelFamilyBuildEntry,
) -> io::Result<()> {
    if model_families
        .iter()
        .any(|existing| existing.source_ordinal == entry.source_ordinal)
    {
        return Err(invalid_data(format!(
            "duplicate model family source ordinal: {}",
            entry.source_ordinal
        )));
    }
    for (field, value, duplicate) in [
        (
            "module",
            entry.module.as_str(),
            model_families
                .iter()
                .any(|existing| existing.module == entry.module),
        ),
        (
            "identifier",
            entry.identifier.as_str(),
            model_families
                .iter()
                .any(|existing| existing.identifier == entry.identifier),
        ),
        (
            "feature id",
            entry.feature_id.as_str(),
            model_families
                .iter()
                .any(|existing| existing.feature_id == entry.feature_id),
        ),
        (
            "fixture",
            entry.fixture.as_str(),
            model_families
                .iter()
                .any(|existing| existing.fixture == entry.fixture),
        ),
    ] {
        if duplicate {
            return Err(invalid_data(format!(
                "duplicate model family {field}: {value}"
            )));
        }
    }
    model_families.push(entry);
    Ok(())
}

pub fn sort_model_family_entries(model_families: &mut [ModelFamilyBuildEntry]) {
    model_families.sort_by_key(|entry| entry.source_ordinal);
}

pub fn model_family_test_names(
    model_families: &[ModelFamilyBuildEntry],
) -> io::Result<Vec<String>> {
    let manifest_directory = env::var_os("CARGO_MANIFEST_DIR").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "Cargo did not provide the CARGO_MANIFEST_DIR build-script variable",
        )
    })?;
    model_family_test_names_in(
        model_families,
        &PathBuf::from(manifest_directory)
            .join("tests")
            .join("families"),
    )
}

pub fn model_family_test_names_in(
    model_families: &[ModelFamilyBuildEntry],
    directory: &std::path::Path,
) -> io::Result<Vec<String>> {
    println!("cargo:rerun-if-changed={}", directory.display());
    let expected = model_families
        .iter()
        .map(|entry| entry.module.clone())
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    if directory.is_dir() {
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            if is_apple_double_metadata(&path)
                || path.extension().and_then(|value| value.to_str()) != Some("rs")
            {
                continue;
            }
            let name = module_name(&path)?;
            if !actual.insert(name.to_owned()) {
                return Err(invalid_data(format!(
                    "duplicate model family test module name: {name}"
                )));
            }
        }
    }
    if actual != expected {
        let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
        let orphaned = actual.difference(&expected).cloned().collect::<Vec<_>>();
        return Err(invalid_data(format!(
            "model family source/test manifest mismatch; missing tests: {missing:?}; orphaned tests: {orphaned:?}"
        )));
    }
    Ok(model_families
        .iter()
        .map(|entry| entry.module.clone())
        .collect())
}

fn source_u16_field(source: &str, path: &std::path::Path, field: &str) -> io::Result<u16> {
    let prefix = format!("{field}:");
    let value = source
        .lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix(&prefix))
        .and_then(|value| value.strip_suffix(','))
        .map(str::trim)
        .ok_or_else(|| {
            invalid_data(format!(
                "model family registration must declare {field}: {}",
                path.display()
            ))
        })?;
    value.parse::<u16>().map_err(|error| {
        invalid_data(format!(
            "model family registration {field} is invalid in {}: {error}",
            path.display()
        ))
    })
}

fn validate_model_family_identity(
    identifier: &str,
    feature_id: &str,
    fixture: &str,
) -> io::Result<()> {
    if identifier.is_empty() || identifier.chars().any(char::is_control) {
        return Err(invalid_data(format!(
            "model family identifier is invalid: {identifier}"
        )));
    }
    let feature_suffix = feature_id
        .strip_prefix("COMFY-MODEL-")
        .ok_or_else(|| invalid_data(format!("model family feature id is invalid: {feature_id}")))?;
    if feature_suffix.len() != 4 || !feature_suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid_data(format!(
            "model family feature id is invalid: {feature_id}"
        )));
    }
    if fixture.is_empty()
        || !fixture
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        || !fixture
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_lowercase)
    {
        return Err(invalid_data(format!(
            "model family fixture is invalid: {fixture}"
        )));
    }
    Ok(())
}

fn invalid_data(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

pub fn model_family_fixture_names(
    model_families: &[(String, String, String, String)],
) -> io::Result<Vec<String>> {
    let manifest_directory = env::var_os("CARGO_MANIFEST_DIR").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "Cargo did not provide the CARGO_MANIFEST_DIR build-script variable",
        )
    })?;
    model_family_fixture_names_in(
        model_families,
        &PathBuf::from(manifest_directory)
            .join("..")
            .join("sim_test_support")
            .join("fixtures")
            .join("models"),
    )
}

pub fn model_family_fixture_names_in(
    model_families: &[(String, String, String, String)],
    directory: &std::path::Path,
) -> io::Result<Vec<String>> {
    println!("cargo:rerun-if-changed={}", directory.display());
    let expected = model_families
        .iter()
        .map(|(_, _, _, fixture)| fixture.to_owned())
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    if directory.is_dir() {
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            if path.is_dir() && path.join("family.json").is_file() {
                let name = path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "model family fixture path is not valid UTF-8",
                        )
                    })?;
                if !actual.insert(name.to_owned()) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("duplicate model family fixture: {name}"),
                    ));
                }
            }
        }
    }
    if actual != expected {
        let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
        let orphaned = actual.difference(&expected).cloned().collect::<Vec<_>>();
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "model family source/fixture manifest mismatch; missing fixtures: {missing:?}; orphaned fixtures: {orphaned:?}"
            ),
        ));
    }
    Ok(actual.into_iter().collect())
}

pub fn register_model_family(
    model_families: &mut Vec<(String, String, String, String)>,
    name: &str,
    identifier: &str,
    feature_id: &str,
    fixture: &str,
) -> io::Result<()> {
    if !valid_module_name(name) {
        return Err(invalid_data(format!(
            "model family module name is invalid: {name}"
        )));
    }
    validate_model_family_identity(identifier, feature_id, fixture)?;
    if model_families
        .iter()
        .any(|(existing, _, _, _)| existing == name)
    {
        return Err(invalid_data(format!(
            "duplicate model family module: {name}"
        )));
    }
    if model_families
        .iter()
        .any(|(_, existing, _, _)| existing == identifier)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("duplicate model family identifier: {identifier}"),
        ));
    }
    if model_families
        .iter()
        .any(|(_, _, existing, _)| existing == feature_id)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("duplicate model family feature id: {feature_id}"),
        ));
    }
    if model_families
        .iter()
        .any(|(_, _, _, existing)| existing == fixture)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("duplicate model family fixture: {fixture}"),
        ));
    }
    model_families.push((
        name.to_owned(),
        identifier.to_owned(),
        feature_id.to_owned(),
        fixture.to_owned(),
    ));
    Ok(())
}

pub fn source_constant<'a>(
    source: &'a str,
    path: &std::path::Path,
    constant: &str,
) -> io::Result<&'a str> {
    let prefix = format!("pub const {constant}: &str = \"");
    let line = source
        .lines()
        .find(|line| line.starts_with(&prefix))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "model family module must declare {constant}: {}",
                    path.display()
                ),
            )
        })?;
    line.strip_prefix(&prefix)
        .and_then(|value| value.strip_suffix("\";"))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "model family constant {constant} is invalid: {}",
                    path.display()
                ),
            )
        })
}

pub fn latent_format_test_names(latent_formats: &[(String, String)]) -> io::Result<Vec<String>> {
    let manifest_directory = env::var_os("CARGO_MANIFEST_DIR").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "Cargo did not provide the CARGO_MANIFEST_DIR build-script variable",
        )
    })?;
    latent_format_test_names_in(
        latent_formats,
        &PathBuf::from(manifest_directory)
            .join("tests")
            .join("latent_formats"),
    )
}

pub fn latent_format_test_names_in(
    latent_formats: &[(String, String)],
    directory: &std::path::Path,
) -> io::Result<Vec<String>> {
    println!("cargo:rerun-if-changed={}", directory.display());
    let expected = latent_formats
        .iter()
        .map(|(name, _)| name.to_owned())
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    if directory.is_dir() {
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            if is_apple_double_metadata(&path)
                || path.extension().and_then(|value| value.to_str()) != Some("rs")
            {
                continue;
            }
            let name = module_name(&path)?;
            if !actual.insert(name.to_owned()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("duplicate latent format test module name: {name}"),
                ));
            }
        }
    }
    if actual != expected {
        let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
        let orphaned = actual.difference(&expected).cloned().collect::<Vec<_>>();
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "latent format source/test manifest mismatch; missing tests: {missing:?}; orphaned tests: {orphaned:?}"
            ),
        ));
    }
    Ok(actual.into_iter().collect())
}

fn is_apple_double_metadata(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("._"))
}

pub fn register_latent_format(
    latent_formats: &mut Vec<(String, String)>,
    name: &str,
    identifier: &str,
) -> io::Result<()> {
    if latent_formats
        .iter()
        .any(|(_, existing)| existing == identifier)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("duplicate latent format identifier: {identifier}"),
        ));
    }
    latent_formats.push((name.to_owned(), identifier.to_owned()));
    Ok(())
}

pub fn latent_format_identifier<'a>(
    source: &'a str,
    path: &std::path::Path,
) -> io::Result<&'a str> {
    const PREFIX: &str = "pub const LATENT_FORMAT_IDENTIFIER: &str = \"";
    let line = source
        .lines()
        .find(|line| line.starts_with(PREFIX))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "latent format module must declare LATENT_FORMAT_IDENTIFIER: {}",
                    path.display()
                ),
            )
        })?;
    let identifier = line
        .strip_prefix(PREFIX)
        .and_then(|value| value.strip_suffix("\";"))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "latent format identifier declaration is invalid: {}",
                    path.display()
                ),
            )
        })?;
    if identifier.is_empty()
        || !identifier
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("latent format identifier is invalid: {identifier}"),
        ));
    }
    Ok(identifier)
}

fn module_name(path: &std::path::Path) -> io::Result<&str> {
    let name = path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("model module path is not valid UTF-8: {}", path.display()),
            )
        })?;
    if !valid_module_name(name) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("model module name is invalid: {name}"),
        ));
    }
    Ok(name)
}

pub fn valid_module_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        && name.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
}
