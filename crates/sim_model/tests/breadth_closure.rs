use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_model::{
    GENERATED_LATENT_FORMAT_MANIFEST, GENERATED_LATENT_FORMATS, GENERATED_MODEL_FAMILY_FIXTURES,
    GENERATED_MODEL_FAMILY_IDENTIFIERS, GENERATED_MODEL_FAMILY_REGISTRATIONS,
    GENERATED_MODEL_FAMILY_SOURCE_MANIFEST, LatentFormatRegistry, ModelFamilyRegistry,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const EXPECTED_FAMILY_COUNT: usize = 94;
const VALIDATION_ID: &str = "VAL-MODEL-FAMILY-001";
const UPDATE_ENV: &str = "UPDATE_SIM_MODEL_FAMILY_CLOSURE";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NativeModelFamilyClosure {
    schema_version: u8,
    validation_id: String,
    generator: String,
    family_count: usize,
    inputs: BTreeMap<String, String>,
    rows: Vec<NativeModelFamilyClosureRow>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NativeModelFamilyClosureRow {
    source_ordinal: u16,
    module: String,
    feature_id: String,
    identifier: String,
    fixture: String,
    source_projection_sha256: String,
    production_sha256: String,
    test_sha256: String,
    fixture_sha256: String,
    provenance_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceCatalog {
    schema_version: u8,
    generator: String,
    inputs: serde_json::Value,
    normalization: serde_json::Value,
    source_ordinal_base: u16,
    model_count: usize,
    models: Vec<SourceModel>,
}

#[derive(Deserialize)]
struct SourceModel {
    feature_id: String,
    name: String,
    source_ordinal: u16,
}

#[derive(Deserialize)]
struct Provenance {
    feature_id: String,
    source_symbol: String,
    source_ordinal: u16,
    source_projection: String,
    source_projection_sha256: String,
}

#[test]
fn val_model_family_001_exact_native_breadth_closure() -> Result<(), Box<dyn std::error::Error>> {
    let expected = build_expected_closure()?;
    validate_closure(&expected)?;
    let serialized = stable_json(&expected)?;
    let path = repository_root()
        .join(".agents/specs/comfy-parity/catalogs/native-model-family-closure.json");
    if std::env::var_os(UPDATE_ENV).is_some_and(|value| value == "1") {
        std::fs::write(&path, &serialized)?;
    }
    let checked_in = std::fs::read(&path)?;
    assert_eq!(
        checked_in, serialized,
        "native model-family closure is stale; rerun with {UPDATE_ENV}=1"
    );
    let decoded: NativeModelFamilyClosure = serde_json::from_slice(&checked_in)?;
    assert_eq!(decoded, expected);
    Ok(())
}

#[test]
fn val_model_family_001_rejects_duplicate_or_partial_closure_before_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let valid = build_expected_closure()?;
    let mut duplicate = valid.clone();
    duplicate.rows[1].feature_id = duplicate.rows[0].feature_id.clone();
    assert!(validate_closure(&duplicate).is_err());

    let mut partial = valid;
    partial.rows.pop();
    partial.family_count -= 1;
    assert!(validate_closure(&partial).is_err());
    Ok(())
}

#[test]
fn val_latent_001_exact_native_breadth_closure() -> Result<(), Box<dyn std::error::Error>> {
    let registry = LatentFormatRegistry::checked(GENERATED_LATENT_FORMATS)?;
    assert_eq!(registry.len(), 33);
    assert_eq!(GENERATED_LATENT_FORMAT_MANIFEST.len(), 33);
    let root = repository_root();
    let mut modules = BTreeSet::new();
    let mut feature_ids = BTreeSet::new();
    let mut identifiers = BTreeSet::new();
    for (module, definition) in GENERATED_LATENT_FORMAT_MANIFEST {
        assert!(modules.insert((*module).to_owned()));
        assert!(feature_ids.insert(definition.feature_id));
        assert!(identifiers.insert(definition.identifier));
        assert!(
            root.join(format!("crates/sim_model/src/latent_formats/{module}.rs"))
                .is_file(),
        );
        assert!(
            root.join(format!("crates/sim_model/tests/latent_formats/{module}.rs"))
                .is_file(),
        );
    }
    let source_modules = rust_module_names(&root.join("crates/sim_model/src/latent_formats"))?;
    let test_modules = rust_module_names(&root.join("crates/sim_model/tests/latent_formats"))?;
    assert_eq!(modules, source_modules);
    assert_eq!(modules, test_modules);
    assert_eq!(modules.len(), registry.len());
    assert_eq!(feature_ids.len(), registry.len());
    assert_eq!(identifiers.len(), registry.len());
    Ok(())
}

fn build_expected_closure() -> Result<NativeModelFamilyClosure, Box<dyn std::error::Error>> {
    let root = repository_root();
    let backend_path = root.join(".agents/specs/comfy-parity/catalogs/backend-models.csv");
    let source_catalog_path = root.join("crates/sim_model/catalog/model-families-v1.json");
    let source_catalog_bytes = std::fs::read(&source_catalog_path)?;
    let source_catalog: SourceCatalog = serde_json::from_slice(&source_catalog_bytes)?;
    assert_eq!(source_catalog.schema_version, 1);
    assert!(!source_catalog.generator.is_empty());
    assert!(source_catalog.inputs.is_array());
    assert!(source_catalog.normalization.is_object());
    assert_eq!(source_catalog.source_ordinal_base, 0);
    assert_eq!(source_catalog.model_count, EXPECTED_FAMILY_COUNT);
    assert_eq!(source_catalog.models.len(), EXPECTED_FAMILY_COUNT);

    let backend_bytes = std::fs::read(&backend_path)?;
    let backend_symbols = backend_model_symbols(&backend_bytes)?;
    assert_eq!(backend_symbols.len(), EXPECTED_FAMILY_COUNT);
    assert_eq!(
        GENERATED_MODEL_FAMILY_SOURCE_MANIFEST.len(),
        EXPECTED_FAMILY_COUNT
    );
    assert_eq!(
        GENERATED_MODEL_FAMILY_IDENTIFIERS.len(),
        EXPECTED_FAMILY_COUNT
    );
    assert_eq!(GENERATED_MODEL_FAMILY_FIXTURES.len(), EXPECTED_FAMILY_COUNT);
    assert_eq!(
        GENERATED_MODEL_FAMILY_REGISTRATIONS.len(),
        EXPECTED_FAMILY_COUNT
    );
    assert_eq!(
        ModelFamilyRegistry::checked_registrations(GENERATED_MODEL_FAMILY_REGISTRATIONS)?.len(),
        EXPECTED_FAMILY_COUNT
    );

    let production_modules = rust_module_names(&root.join("crates/sim_model/src/families"))?;
    let test_modules = rust_module_names(&root.join("crates/sim_model/tests/families"))?;
    let fixture_names = model_fixture_names(&root.join("crates/sim_test_support/fixtures/models"))?;
    assert_eq!(production_modules.len(), EXPECTED_FAMILY_COUNT);
    assert_eq!(production_modules, test_modules);
    assert_eq!(fixture_names.len(), EXPECTED_FAMILY_COUNT);

    let mut rows = Vec::with_capacity(EXPECTED_FAMILY_COUNT);
    for (index, (((source_entry, identifier), registration), source_model)) in
        GENERATED_MODEL_FAMILY_SOURCE_MANIFEST
            .iter()
            .zip(GENERATED_MODEL_FAMILY_IDENTIFIERS)
            .zip(GENERATED_MODEL_FAMILY_REGISTRATIONS)
            .zip(&source_catalog.models)
            .enumerate()
    {
        let (module, feature_id, fixture, source_ordinal) = *source_entry;
        let expected_ordinal = u16::try_from(index)?;
        assert_eq!(source_ordinal, expected_ordinal);
        assert_eq!(registration.source_ordinal, source_ordinal);
        assert_eq!(registration.definition.feature_id, feature_id);
        assert_eq!(registration.definition.identifier, *identifier);
        assert_eq!(source_model.source_ordinal, source_ordinal);
        assert_eq!(source_model.feature_id, feature_id);
        assert_eq!(source_model.name, *identifier);
        assert_eq!(
            backend_symbols.get(feature_id).map(String::as_str),
            Some(*identifier)
        );
        assert!(production_modules.contains(module));
        assert!(test_modules.contains(module));
        assert!(fixture_names.contains(fixture));

        let production_path = root.join(format!("crates/sim_model/src/families/{module}.rs"));
        let test_path = root.join(format!("crates/sim_model/tests/families/{module}.rs"));
        let fixture_path = root.join(format!(
            "crates/sim_test_support/fixtures/models/{fixture}/family.json"
        ));
        let provenance_path = root.join(format!(
            "crates/sim_test_support/fixtures/models/{fixture}/provenance.json"
        ));
        let provenance_bytes = std::fs::read(&provenance_path)?;
        let provenance: Provenance = serde_json::from_slice(&provenance_bytes)?;
        assert_eq!(provenance.feature_id, feature_id);
        assert_eq!(provenance.source_symbol, *identifier);
        assert_eq!(provenance.source_ordinal, source_ordinal);
        assert_eq!(
            sha256(provenance.source_projection.as_bytes()),
            provenance.source_projection_sha256
        );

        rows.push(NativeModelFamilyClosureRow {
            source_ordinal,
            module: module.to_owned(),
            feature_id: feature_id.to_owned(),
            identifier: (*identifier).to_owned(),
            fixture: fixture.to_owned(),
            source_projection_sha256: provenance.source_projection_sha256,
            production_sha256: sha256_file(&production_path)?,
            test_sha256: sha256_file(&test_path)?,
            fixture_sha256: sha256_file(&fixture_path)?,
            provenance_sha256: sha256(&provenance_bytes),
        });
    }

    let inputs = [
        ("backend-models.csv", sha256(&backend_bytes)),
        ("model-families-v1.json", sha256(&source_catalog_bytes)),
        (
            "crates/sim_model/build.rs",
            sha256_file(&root.join("crates/sim_model/build.rs"))?,
        ),
        (
            "crates/sim_model/tests/model_families.rs",
            sha256_file(&root.join("crates/sim_model/tests/model_families.rs"))?,
        ),
    ]
    .into_iter()
    .map(|(path, digest)| (path.to_owned(), digest))
    .collect();

    Ok(NativeModelFamilyClosure {
        schema_version: 1,
        validation_id: VALIDATION_ID.to_owned(),
        generator: "crates/sim_model/tests/breadth_closure.rs".to_owned(),
        family_count: rows.len(),
        inputs,
        rows,
    })
}

fn validate_closure(closure: &NativeModelFamilyClosure) -> Result<(), String> {
    if closure.schema_version != 1
        || closure.validation_id != VALIDATION_ID
        || closure.generator.is_empty()
        || closure.family_count != EXPECTED_FAMILY_COUNT
        || closure.rows.len() != EXPECTED_FAMILY_COUNT
        || closure.inputs.values().any(|digest| !valid_sha256(digest))
    {
        return Err("model-family closure header is incomplete".to_owned());
    }
    let mut modules = BTreeSet::new();
    let mut features = BTreeSet::new();
    let mut identifiers = BTreeSet::new();
    let mut fixtures = BTreeSet::new();
    for (index, row) in closure.rows.iter().enumerate() {
        if row.source_ordinal != u16::try_from(index).map_err(|error| error.to_string())?
            || !modules.insert(&row.module)
            || !features.insert(&row.feature_id)
            || !identifiers.insert(&row.identifier)
            || !fixtures.insert(&row.fixture)
            || [
                &row.source_projection_sha256,
                &row.production_sha256,
                &row.test_sha256,
                &row.fixture_sha256,
                &row.provenance_sha256,
            ]
            .into_iter()
            .any(|digest| !valid_sha256(digest))
        {
            return Err(format!("invalid or duplicate closure row at index {index}"));
        }
    }
    Ok(())
}

fn backend_model_symbols(
    bytes: &[u8],
) -> Result<BTreeMap<String, String>, Box<dyn std::error::Error>> {
    let text = std::str::from_utf8(bytes)?;
    let rows = parse_csv(text)?;
    let header = rows.first().ok_or("backend model catalog is empty")?;
    let feature_index = column(header, "feature_id")?;
    let symbol_index = column(header, "source_symbol")?;
    let mut result = BTreeMap::new();
    for row in rows.iter().skip(1) {
        if row.len() != header.len() {
            return Err("backend model catalog row width mismatch".into());
        }
        let Some(number) = row[feature_index]
            .strip_prefix("COMFY-MODEL-")
            .and_then(|number| number.parse::<u16>().ok())
        else {
            continue;
        };
        if (61..=154).contains(&number)
            && result
                .insert(row[feature_index].clone(), row[symbol_index].clone())
                .is_some()
        {
            return Err(format!("duplicate backend model feature: {}", row[feature_index]).into());
        }
    }
    Ok(result)
}

fn parse_csv(text: &str) -> Result<Vec<Vec<String>>, Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        match character {
            '"' if quoted && characters.peek() == Some(&'"') => {
                field.push('"');
                characters.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => row.push(std::mem::take(&mut field)),
            '\n' if !quoted => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            '\r' if !quoted => {}
            _ => field.push(character),
        }
    }
    if quoted {
        return Err("unterminated quoted CSV field".into());
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    Ok(rows)
}

fn column(header: &[String], name: &str) -> Result<usize, Box<dyn std::error::Error>> {
    header
        .iter()
        .position(|column| column == name)
        .ok_or_else(|| format!("backend model catalog is missing {name}").into())
}

fn rust_module_names(directory: &Path) -> std::io::Result<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();
        if ignored_path(&path) || path.extension().and_then(|value| value.to_str()) != Some("rs") {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| std::io::Error::other("non-UTF-8 model-family module"))?;
        if !names.insert(name.to_owned()) {
            return Err(std::io::Error::other(format!(
                "duplicate model-family module: {name}"
            )));
        }
    }
    Ok(names)
}

fn model_fixture_names(directory: &Path) -> std::io::Result<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();
        if ignored_path(&path) || !path.join("family.json").is_file() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| std::io::Error::other("non-UTF-8 model-family fixture"))?;
        if !names.insert(name.to_owned()) {
            return Err(std::io::Error::other(format!(
                "duplicate model-family fixture: {name}"
            )));
        }
    }
    Ok(names)
}

fn ignored_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|name| name.starts_with("._"))
}

fn stable_json(value: &NativeModelFamilyClosure) -> Result<Vec<u8>, serde_json::Error> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    Ok(sha256(&std::fs::read(path)?))
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
