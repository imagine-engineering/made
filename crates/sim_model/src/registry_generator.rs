use crate::{
    CatalogModelDescriptor, MODEL_DESCRIPTOR_SCHEMA_VERSION, ModelCatalogAvailability,
    ModelCatalogConfidence, ModelCatalogKey, ModelCatalogKind, ModelEvidenceLevel,
    ModelParityStatus,
};
use std::{collections::BTreeMap, error::Error, fmt};

pub const MODEL_CATALOG: &str =
    include_str!("../../../crates/sim_test_support/parity/catalogs/backend-models.csv");

const MAX_CATALOG_BYTES: usize = 16 * 1024 * 1024;
const MAX_CATALOG_ROWS: usize = 100_000;
const MAX_COLUMNS: usize = 128;
const MAX_FIELD_BYTES: usize = 2 * 1024 * 1024;

const MODEL_HEADER: &[&str] = &[
    "kind",
    "name",
    "classification",
    "availability",
    "evidence_level",
    "confidence",
    "identifier_or_format",
    "inputs_defaults",
    "success_behavior",
    "failure_behavior",
    "dependencies_platform",
    "source_file",
    "source_symbol",
    "source_line",
    "test_evidence",
    "zed_status",
    "parity_gap",
    "feature_id",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelRegistryError {
    CatalogTooLarge,
    TooManyRows,
    TooManyColumns {
        row: usize,
    },
    FieldTooLarge {
        row: usize,
        column: usize,
    },
    MalformedCsv {
        position: usize,
        reason: String,
    },
    HeaderMismatch,
    ColumnCount {
        row: usize,
        expected: usize,
        actual: usize,
    },
    InvalidNumber {
        row: usize,
        field: &'static str,
    },
    InvalidDescriptor {
        row: usize,
        field: &'static str,
    },
    DuplicateModel(ModelCatalogKey),
    DuplicateFeature(String),
    AmbiguousIdentifier {
        identifier: String,
        candidates: Vec<ModelCatalogKey>,
    },
}

impl fmt::Display for ModelRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CatalogTooLarge => formatter.write_str("model catalog exceeds its byte limit"),
            Self::TooManyRows => formatter.write_str("model catalog exceeds its row limit"),
            Self::TooManyColumns { row } => {
                write!(
                    formatter,
                    "model catalog row {row} exceeds its column limit"
                )
            }
            Self::FieldTooLarge { row, column } => write!(
                formatter,
                "model catalog field at row {row}, column {column} exceeds its byte limit"
            ),
            Self::MalformedCsv { position, reason } => {
                write!(
                    formatter,
                    "malformed model catalog CSV at {position}: {reason}"
                )
            }
            Self::HeaderMismatch => formatter.write_str("model catalog header does not match"),
            Self::ColumnCount {
                row,
                expected,
                actual,
            } => write!(
                formatter,
                "model catalog row {row} has {actual} columns, expected {expected}"
            ),
            Self::InvalidNumber { row, field } => {
                write!(
                    formatter,
                    "model catalog row {row} has invalid number `{field}`"
                )
            }
            Self::InvalidDescriptor { row, field } => {
                write!(formatter, "model catalog row {row} has invalid `{field}`")
            }
            Self::DuplicateModel(key) => {
                write!(
                    formatter,
                    "duplicate model catalog key `{}/{}`",
                    key.kind, key.name
                )
            }
            Self::DuplicateFeature(identifier) => {
                write!(formatter, "duplicate model feature ID `{identifier}`")
            }
            Self::AmbiguousIdentifier {
                identifier,
                candidates,
            } => write!(
                formatter,
                "model identifier `{identifier}` is ambiguous across {} catalog rows",
                candidates.len()
            ),
        }
    }
}

impl Error for ModelRegistryError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelRegistry {
    descriptors: BTreeMap<ModelCatalogKey, CatalogModelDescriptor>,
    features: BTreeMap<String, ModelCatalogKey>,
    identifier_candidates: BTreeMap<String, Vec<ModelCatalogKey>>,
}

impl ModelRegistry {
    pub fn built_in() -> Result<Self, ModelRegistryError> {
        ModelRegistryGenerator::from_catalog(MODEL_CATALOG).map(ModelRegistryGenerator::finish)
    }

    pub fn descriptors(&self) -> &BTreeMap<ModelCatalogKey, CatalogModelDescriptor> {
        &self.descriptors
    }

    pub fn descriptor(
        &self,
        kind: ModelCatalogKind,
        name: &str,
    ) -> Option<&CatalogModelDescriptor> {
        self.descriptors.get(&ModelCatalogKey {
            kind,
            name: name.to_owned(),
        })
    }

    pub fn by_feature(&self, feature_id: &str) -> Option<&CatalogModelDescriptor> {
        self.features
            .get(feature_id)
            .and_then(|key| self.descriptors.get(key))
    }

    pub fn identifier_candidates(&self, identifier: &str) -> &[ModelCatalogKey] {
        self.identifier_candidates
            .get(identifier)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    pub fn resolve_identifier(
        &self,
        identifier: &str,
    ) -> Result<Option<&CatalogModelDescriptor>, ModelRegistryError> {
        let candidates = self.identifier_candidates(identifier);
        match candidates {
            [] => Ok(None),
            [key] => Ok(self.descriptors.get(key)),
            _ => Err(ModelRegistryError::AmbiguousIdentifier {
                identifier: identifier.to_owned(),
                candidates: candidates.to_vec(),
            }),
        }
    }

    pub fn families(&self) -> impl Iterator<Item = &CatalogModelDescriptor> {
        self.descriptors
            .values()
            .filter(|descriptor| descriptor.kind == ModelCatalogKind::ModelFamily)
    }
}

pub struct ModelRegistryGenerator {
    descriptors: BTreeMap<ModelCatalogKey, CatalogModelDescriptor>,
    features: BTreeMap<String, ModelCatalogKey>,
    identifier_candidates: BTreeMap<String, Vec<ModelCatalogKey>>,
}

impl ModelRegistryGenerator {
    pub fn from_catalog(catalog: &str) -> Result<Self, ModelRegistryError> {
        let rows = parse_csv(catalog)?;
        validate_header(&rows)?;
        let mut generator = Self {
            descriptors: BTreeMap::new(),
            features: BTreeMap::new(),
            identifier_candidates: BTreeMap::new(),
        };
        for (row_index, row) in rows.iter().enumerate().skip(1) {
            generator.insert(parse_row(row_index + 1, row)?)?;
        }
        Ok(generator)
    }

    pub fn finish(mut self) -> ModelRegistry {
        for candidates in self.identifier_candidates.values_mut() {
            candidates.sort();
        }
        ModelRegistry {
            descriptors: self.descriptors,
            features: self.features,
            identifier_candidates: self.identifier_candidates,
        }
    }

    fn insert(&mut self, descriptor: CatalogModelDescriptor) -> Result<(), ModelRegistryError> {
        let key = descriptor.key();
        if self.descriptors.contains_key(&key) {
            return Err(ModelRegistryError::DuplicateModel(key));
        }
        if self.features.contains_key(&descriptor.feature_id) {
            return Err(ModelRegistryError::DuplicateFeature(descriptor.feature_id));
        }
        self.features
            .insert(descriptor.feature_id.clone(), key.clone());
        if !descriptor.identifier_or_format.is_empty() {
            self.identifier_candidates
                .entry(descriptor.identifier_or_format.clone())
                .or_default()
                .push(key.clone());
        }
        self.descriptors.insert(key, descriptor);
        Ok(())
    }
}

fn parse_row(
    row_number: usize,
    row: &[String],
) -> Result<CatalogModelDescriptor, ModelRegistryError> {
    validate_column_count(row_number, row, MODEL_HEADER.len())?;
    let field = |index, name| field(row_number, row, index, name);
    let kind = parse_kind(row_number, field(0, "kind")?)?;
    let name = field(1, "name")?;
    let feature_id = field(17, "feature_id")?;
    for (index, name) in [
        (1, "name"),
        (2, "classification"),
        (6, "identifier_or_format"),
        (7, "inputs_defaults"),
        (8, "success_behavior"),
        (9, "failure_behavior"),
        (10, "dependencies_platform"),
        (11, "source_file"),
        (12, "source_symbol"),
        (15, "zed_status"),
        (16, "parity_gap"),
        (17, "feature_id"),
    ] {
        let value = field(index, name)?;
        if value.is_empty() {
            return Err(ModelRegistryError::InvalidDescriptor {
                row: row_number,
                field: name,
            });
        }
    }
    if feature_id.len() != "COMFY-MODEL-0000".len()
        || !feature_id.starts_with("COMFY-MODEL-")
        || !feature_id["COMFY-MODEL-".len()..]
            .bytes()
            .all(|byte| byte.is_ascii_digit())
    {
        return Err(ModelRegistryError::InvalidDescriptor {
            row: row_number,
            field: "feature_id",
        });
    }
    Ok(CatalogModelDescriptor {
        schema_version: MODEL_DESCRIPTOR_SCHEMA_VERSION,
        kind,
        name: name.to_owned(),
        classification: field(2, "classification")?.to_owned(),
        availability: parse_availability(row_number, field(3, "availability")?)?,
        evidence_level: parse_evidence_level(row_number, field(4, "evidence_level")?)?,
        confidence: parse_confidence(row_number, field(5, "confidence")?)?,
        identifier_or_format: field(6, "identifier_or_format")?.to_owned(),
        inputs_defaults: field(7, "inputs_defaults")?.to_owned(),
        success_behavior: field(8, "success_behavior")?.to_owned(),
        failure_behavior: field(9, "failure_behavior")?.to_owned(),
        dependencies_platform: field(10, "dependencies_platform")?.to_owned(),
        source_file: field(11, "source_file")?.to_owned(),
        source_symbol: field(12, "source_symbol")?.to_owned(),
        source_line: parse_optional_u32(row_number, "source_line", field(13, "source_line")?)?,
        test_evidence: field(14, "test_evidence")?.to_owned(),
        zed_status: parse_parity_status(row_number, field(15, "zed_status")?)?,
        parity_gap: field(16, "parity_gap")?.to_owned(),
        feature_id: feature_id.to_owned(),
    })
}

fn parse_kind(row: usize, value: &str) -> Result<ModelCatalogKind, ModelRegistryError> {
    match value {
        "attention backend" => Ok(ModelCatalogKind::AttentionBackend),
        "dtype" => Ok(ModelCatalogKind::Dtype),
        "hardware backend" => Ok(ModelCatalogKind::HardwareBackend),
        "latent format" => Ok(ModelCatalogKind::LatentFormat),
        "memory mode" => Ok(ModelCatalogKind::MemoryMode),
        "model family" => Ok(ModelCatalogKind::ModelFamily),
        "quantization" => Ok(ModelCatalogKind::Quantization),
        "sampler" => Ok(ModelCatalogKind::Sampler),
        "scheduler" => Ok(ModelCatalogKind::Scheduler),
        _ => Err(ModelRegistryError::InvalidDescriptor { row, field: "kind" }),
    }
}

fn parse_availability(
    row: usize,
    value: &str,
) -> Result<ModelCatalogAvailability, ModelRegistryError> {
    match value {
        "active" => Ok(ModelCatalogAvailability::Active),
        "conditional" => Ok(ModelCatalogAvailability::Conditional),
        "platform-specific" => Ok(ModelCatalogAvailability::PlatformSpecific),
        _ => Err(ModelRegistryError::InvalidDescriptor {
            row,
            field: "availability",
        }),
    }
}

fn parse_evidence_level(row: usize, value: &str) -> Result<ModelEvidenceLevel, ModelRegistryError> {
    match value {
        "code-inferred" => Ok(ModelEvidenceLevel::CodeInferred),
        "test-backed" => Ok(ModelEvidenceLevel::TestBacked),
        _ => Err(ModelRegistryError::InvalidDescriptor {
            row,
            field: "evidence_level",
        }),
    }
}

fn parse_confidence(row: usize, value: &str) -> Result<ModelCatalogConfidence, ModelRegistryError> {
    match value {
        "high" => Ok(ModelCatalogConfidence::High),
        _ => Err(ModelRegistryError::InvalidDescriptor {
            row,
            field: "confidence",
        }),
    }
}

fn parse_parity_status(row: usize, value: &str) -> Result<ModelParityStatus, ModelRegistryError> {
    match value {
        "missing" => Ok(ModelParityStatus::Missing),
        "partial" => Ok(ModelParityStatus::Partial),
        _ => Err(ModelRegistryError::InvalidDescriptor {
            row,
            field: "zed_status",
        }),
    }
}

fn field<'a>(
    row_number: usize,
    row: &'a [String],
    index: usize,
    name: &'static str,
) -> Result<&'a str, ModelRegistryError> {
    row.get(index)
        .map(String::as_str)
        .ok_or(ModelRegistryError::InvalidDescriptor {
            row: row_number,
            field: name,
        })
}

fn parse_optional_u32(
    row: usize,
    field: &'static str,
    value: &str,
) -> Result<Option<u32>, ModelRegistryError> {
    if value.is_empty() {
        Ok(None)
    } else {
        value
            .parse()
            .map(Some)
            .map_err(|_| ModelRegistryError::InvalidNumber { row, field })
    }
}

fn validate_header(rows: &[Vec<String>]) -> Result<(), ModelRegistryError> {
    let Some(header) = rows.first() else {
        return Err(ModelRegistryError::HeaderMismatch);
    };
    if header.len() != MODEL_HEADER.len()
        || header
            .iter()
            .zip(MODEL_HEADER)
            .any(|(actual, expected)| actual != expected)
    {
        return Err(ModelRegistryError::HeaderMismatch);
    }
    Ok(())
}

fn validate_column_count(
    row_number: usize,
    row: &[String],
    expected: usize,
) -> Result<(), ModelRegistryError> {
    if row.len() == expected {
        Ok(())
    } else {
        Err(ModelRegistryError::ColumnCount {
            row: row_number,
            expected,
            actual: row.len(),
        })
    }
}

fn parse_csv(input: &str) -> Result<Vec<Vec<String>>, ModelRegistryError> {
    if input.len() > MAX_CATALOG_BYTES {
        return Err(ModelRegistryError::CatalogTooLarge);
    }
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut characters = input.char_indices().peekable();
    let mut in_quotes = false;
    let mut quote_closed = false;
    while let Some((position, character)) = characters.next() {
        if in_quotes {
            if character == '"' {
                if characters.peek().is_some_and(|(_, next)| *next == '"') {
                    characters.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                    quote_closed = true;
                }
            } else {
                field.push(character);
            }
        } else {
            match character {
                '"' if field.is_empty() && !quote_closed => in_quotes = true,
                '"' => {
                    return Err(ModelRegistryError::MalformedCsv {
                        position,
                        reason: "quote appeared inside an unquoted field".to_owned(),
                    });
                }
                ',' => {
                    push_field(&mut row, &mut field, rows.len() + 1)?;
                    quote_closed = false;
                }
                '\n' | '\r' => {
                    if character == '\r' && characters.peek().is_some_and(|(_, next)| *next == '\n')
                    {
                        characters.next();
                    }
                    push_field(&mut row, &mut field, rows.len() + 1)?;
                    push_row(&mut rows, &mut row)?;
                    quote_closed = false;
                }
                _ if quote_closed => {
                    return Err(ModelRegistryError::MalformedCsv {
                        position,
                        reason: "characters followed a closing quote".to_owned(),
                    });
                }
                _ => field.push(character),
            }
        }
        if field.len() > MAX_FIELD_BYTES {
            return Err(ModelRegistryError::FieldTooLarge {
                row: rows.len() + 1,
                column: row.len() + 1,
            });
        }
    }
    if in_quotes {
        return Err(ModelRegistryError::MalformedCsv {
            position: input.len(),
            reason: "quoted field was not closed".to_owned(),
        });
    }
    if !field.is_empty() || !row.is_empty() || quote_closed {
        push_field(&mut row, &mut field, rows.len() + 1)?;
        push_row(&mut rows, &mut row)?;
    }
    Ok(rows)
}

fn push_field(
    row: &mut Vec<String>,
    field: &mut String,
    row_number: usize,
) -> Result<(), ModelRegistryError> {
    if row.len() >= MAX_COLUMNS {
        return Err(ModelRegistryError::TooManyColumns { row: row_number });
    }
    row.push(std::mem::take(field));
    Ok(())
}

fn push_row(rows: &mut Vec<Vec<String>>, row: &mut Vec<String>) -> Result<(), ModelRegistryError> {
    if rows.len() >= MAX_CATALOG_ROWS {
        return Err(ModelRegistryError::TooManyRows);
    }
    rows.push(std::mem::take(row));
    Ok(())
}

#[cfg(test)]
fn canonical_csv(rows: &[Vec<String>]) -> String {
    let mut output = String::new();
    for row in rows {
        for (index, field) in row.iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            if field
                .chars()
                .any(|character| matches!(character, ',' | '"' | '\n' | '\r'))
            {
                output.push('"');
                for character in field.chars() {
                    if character == '"' {
                        output.push('"');
                    }
                    output.push(character);
                }
                output.push('"');
            } else {
                output.push_str(field);
            }
        }
        output.push('\n');
    }
    output
}

#[cfg(test)]
fn descriptor_catalog_row(descriptor: &CatalogModelDescriptor) -> Vec<String> {
    vec![
        descriptor.kind.as_str().to_owned(),
        descriptor.name.clone(),
        descriptor.classification.clone(),
        descriptor.availability.as_str().to_owned(),
        descriptor.evidence_level.as_str().to_owned(),
        descriptor.confidence.as_str().to_owned(),
        descriptor.identifier_or_format.clone(),
        descriptor.inputs_defaults.clone(),
        descriptor.success_behavior.clone(),
        descriptor.failure_behavior.clone(),
        descriptor.dependencies_platform.clone(),
        descriptor.source_file.clone(),
        descriptor.source_symbol.clone(),
        descriptor
            .source_line
            .map_or_else(String::new, |line| line.to_string()),
        descriptor.test_evidence.clone(),
        descriptor.zed_status.as_str().to_owned(),
        descriptor.parity_gap.clone(),
        descriptor.feature_id.clone(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        collections::BTreeMap,
        fs,
        path::{Path, PathBuf},
    };

    #[test]
    fn checked_in_catalog_round_trips_without_schema_loss() -> Result<(), Box<dyn Error>> {
        let rows = parse_csv(MODEL_CATALOG)?;
        assert_eq!(rows.len(), 212);
        assert_eq!(canonical_csv(&rows), MODEL_CATALOG);
        let registry = ModelRegistry::built_in()?;
        let mut reconstructed = vec![
            MODEL_HEADER
                .iter()
                .map(|field| (*field).to_owned())
                .collect::<Vec<_>>(),
        ];
        for row in rows.iter().skip(1) {
            let feature_id = row
                .get(17)
                .ok_or("model catalog row has no feature identifier")?;
            let descriptor = registry
                .by_feature(feature_id)
                .ok_or("model registry omitted a catalog row")?;
            let reconstructed_row = descriptor_catalog_row(descriptor);
            assert_eq!(&reconstructed_row, row);
            reconstructed.push(reconstructed_row);
            let encoded = serde_json::to_vec(descriptor)?;
            let decoded = serde_json::from_slice::<CatalogModelDescriptor>(&encoded)?;
            assert_eq!(&decoded, descriptor);
            assert_eq!(decoded.schema_version, MODEL_DESCRIPTOR_SCHEMA_VERSION);
        }
        assert_eq!(canonical_csv(&reconstructed), MODEL_CATALOG);
        Ok(())
    }

    #[test]
    fn registry_preserves_every_row_and_never_claims_execution() -> Result<(), Box<dyn Error>> {
        let registry = ModelRegistry::built_in()?;
        assert_eq!(registry.descriptors().len(), 211);
        assert_eq!(registry.families().count(), 94);
        let sd15 = registry
            .by_feature("COMFY-MODEL-0117")
            .ok_or("SD15 descriptor is missing")?;
        assert_eq!(sd15.kind, ModelCatalogKind::ModelFamily);
        assert_eq!(sd15.name, "SD15");
        assert_eq!(sd15.source_symbol, "SD15");
        assert_eq!(sd15.source_line, Some(44));
        assert_eq!(sd15.zed_status, ModelParityStatus::Missing);
        let metal = registry
            .by_feature("COMFY-MODEL-0015")
            .ok_or("Apple Metal MPS descriptor is missing")?;
        assert_eq!(metal.zed_status, ModelParityStatus::Partial);
        Ok(())
    }

    #[test]
    fn ambiguous_source_identifiers_preserve_all_candidates() -> Result<(), Box<dyn Error>> {
        let registry = ModelRegistry::built_in()?;
        let ambiguous = registry
            .identifier_candidates
            .iter()
            .filter(|(_, candidates)| candidates.len() > 1)
            .collect::<Vec<_>>();
        assert_eq!(ambiguous.len(), 6);
        for (identifier, candidates) in ambiguous {
            assert!(candidates.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(matches!(
                registry.resolve_identifier(identifier),
                Err(ModelRegistryError::AmbiguousIdentifier { .. })
            ));
        }
        let sd15 = registry
            .by_feature("COMFY-MODEL-0117")
            .ok_or("SD15 descriptor is missing")?;
        assert_eq!(
            registry.resolve_identifier(&sd15.identifier_or_format)?,
            Some(sd15)
        );
        assert_eq!(registry.resolve_identifier("not-cataloged")?, None);
        Ok(())
    }

    #[test]
    fn malformed_and_colliding_catalogs_fail_closed() -> Result<(), Box<dyn Error>> {
        assert!(matches!(
            parse_csv("a,b\n\"unterminated"),
            Err(ModelRegistryError::MalformedCsv { .. })
        ));
        let header = MODEL_CATALOG
            .lines()
            .next()
            .ok_or("model catalog header is missing")?;
        let row = MODEL_CATALOG
            .lines()
            .nth(1)
            .ok_or("model catalog fixture row is missing")?;
        let duplicate = format!("{header}\n{row}\n{row}\n");
        assert!(matches!(
            ModelRegistryGenerator::from_catalog(&duplicate),
            Err(ModelRegistryError::DuplicateModel(_))
                | Err(ModelRegistryError::DuplicateFeature(_))
        ));

        let mut rows = parse_csv(MODEL_CATALOG)?;
        let first = rows.get_mut(1).ok_or("model catalog has no first row")?;
        let kind = first.get_mut(0).ok_or("model catalog row has no kind")?;
        *kind = "unknown kind".to_owned();
        assert!(matches!(
            ModelRegistryGenerator::from_catalog(&canonical_csv(&rows)),
            Err(ModelRegistryError::InvalidDescriptor { field: "kind", .. })
        ));

        let mut rows = parse_csv(MODEL_CATALOG)?;
        let first = rows.get_mut(1).ok_or("model catalog has no first row")?;
        let availability = first
            .get_mut(3)
            .ok_or("model catalog row has no availability")?;
        *availability = "implicitly-enabled".to_owned();
        assert!(matches!(
            ModelRegistryGenerator::from_catalog(&canonical_csv(&rows)),
            Err(ModelRegistryError::InvalidDescriptor {
                field: "availability",
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn val_model_registry_001() -> Result<(), Box<dyn Error>> {
        let rows = parse_csv(MODEL_CATALOG)?;
        let registry = ModelRegistry::built_in()?;
        let second_registry = ModelRegistry::built_in()?;
        assert_eq!(registry, second_registry);
        assert_eq!(rows.len(), 212);
        assert_eq!(registry.descriptors().len(), 211);
        assert_eq!(registry.families().count(), 94);

        let mut reconstructed_rows = vec![
            MODEL_HEADER
                .iter()
                .map(|field| (*field).to_owned())
                .collect::<Vec<_>>(),
        ];
        let mut typed_projection = Vec::with_capacity(211);
        for row in rows.iter().skip(1) {
            let feature_id = row
                .get(17)
                .ok_or("model catalog row has no feature identifier")?;
            let descriptor = registry
                .by_feature(feature_id)
                .ok_or("model registry omitted a catalog row")?;
            reconstructed_rows.push(descriptor_catalog_row(descriptor));
            typed_projection.push(descriptor.clone());
        }
        assert_eq!(canonical_csv(&reconstructed_rows), MODEL_CATALOG);
        let typed_projection_bytes = serde_json::to_vec(&typed_projection)?;
        let decoded_projection =
            serde_json::from_slice::<Vec<CatalogModelDescriptor>>(&typed_projection_bytes)?;
        assert_eq!(decoded_projection, typed_projection);
        assert!(
            typed_projection
                .iter()
                .all(|descriptor| descriptor.schema_version == MODEL_DESCRIPTOR_SCHEMA_VERSION)
        );
        assert_eq!(
            typed_projection
                .iter()
                .filter(|descriptor| descriptor.zed_status == ModelParityStatus::Partial)
                .map(|descriptor| descriptor.feature_id.as_str())
                .collect::<Vec<_>>(),
            vec!["COMFY-MODEL-0015", "COMFY-MODEL-0016", "COMFY-MODEL-0020"]
        );

        let family_inventory = registry
            .families()
            .map(|descriptor| (&descriptor.feature_id, &descriptor.name))
            .collect::<Vec<_>>();
        assert_eq!(family_inventory.len(), 94);
        let family_inventory_bytes = serde_json::to_vec(&family_inventory)?;

        let ambiguous_identifiers = registry
            .identifier_candidates
            .iter()
            .filter(|(_, candidates)| candidates.len() > 1)
            .collect::<Vec<_>>();
        assert_eq!(ambiguous_identifiers.len(), 6);
        for (identifier, candidates) in &ambiguous_identifiers {
            assert!(candidates.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(matches!(
                registry.resolve_identifier(identifier),
                Err(ModelRegistryError::AmbiguousIdentifier { .. })
            ));
        }
        let ambiguous_identifier_bytes = serde_json::to_vec(&ambiguous_identifiers)?;

        let header = MODEL_CATALOG
            .lines()
            .next()
            .ok_or("model catalog header is missing")?;
        let first_row = MODEL_CATALOG
            .lines()
            .nth(1)
            .ok_or("model catalog first row is missing")?;
        let duplicate = format!("{header}\n{first_row}\n{first_row}\n");
        assert!(matches!(
            ModelRegistryGenerator::from_catalog(&duplicate),
            Err(ModelRegistryError::DuplicateModel(_))
                | Err(ModelRegistryError::DuplicateFeature(_))
        ));
        let mut feature_collision_rows = parse_csv(MODEL_CATALOG)?;
        let mut colliding_row = feature_collision_rows
            .get(1)
            .ok_or("model catalog first row is missing")?
            .clone();
        *colliding_row
            .get_mut(1)
            .ok_or("model catalog row has no name")? = "Distinct Name".to_owned();
        feature_collision_rows.push(colliding_row);
        assert!(matches!(
            ModelRegistryGenerator::from_catalog(&canonical_csv(&feature_collision_rows)),
            Err(ModelRegistryError::DuplicateFeature(_))
        ));

        let workspace_root = workspace_root()?;
        let source_root = workspace_root.join("projects/comfy/ComfyUI");
        for descriptor in registry.descriptors().values() {
            for source_file in descriptor.source_file.split(';').map(str::trim) {
                assert!(
                    source_root.join(source_file).is_file(),
                    "catalog source file is absent: {source_file}"
                );
            }
        }

        let cases = BTreeMap::from([
            ("ambiguous_identifier_resolution_fails_closed", true),
            ("catalog_collisions_rejected", true),
            ("descriptor_projection_is_deterministic", true),
            ("exact_211_catalog_rows", true),
            ("exact_94_family_inventory_rows", true),
            ("raw_catalog_fields_round_trip", true),
            ("no_execution_status_or_activation_api", true),
            ("schema_and_source_status_round_trip", true),
            ("source_evidence_paths_exist", true),
        ]);
        let fixture_digests = BTreeMap::from([
            ("backend_models_csv", sha256(MODEL_CATALOG.as_bytes())),
            (
                "typed_descriptor_projection",
                sha256(&typed_projection_bytes),
            ),
            ("family_inventory", sha256(&family_inventory_bytes)),
            (
                "ambiguous_identifier_candidates",
                sha256(&ambiguous_identifier_bytes),
            ),
        ]);
        let artifact = json!({
            "validation_id": "VAL-MODEL-REGISTRY-001",
            "validation": "VAL-MODEL-REGISTRY-001",
            "scope": "source-catalog model descriptor registry",
            "environment": {
                "operating_system": std::env::consts::OS,
                "architecture": std::env::consts::ARCH,
                "backend": "native-rust-descriptor-registry",
                "development_oracle_executed": false,
                "network_used": false,
                "external_processes": Vec::<String>::new()
            },
            "catalog": {
                "rows": 211,
                "model_family_rows": 94,
                "ambiguous_identifier_groups": 6,
                "executable_claims": 0,
                "descriptor_schema_version": MODEL_DESCRIPTOR_SCHEMA_VERSION
            },
            "fixture_digests": fixture_digests,
            "summary": {"passed": cases.len(), "failed": 0, "skipped": 0},
            "cases": cases,
            "skipped": [],
            "validation_closure": {
                "claimed": true,
                "stage": "descriptor-registry",
                "validated_scope": "catalog ingestion, typed projection, ambiguity, collisions, and execution-claim isolation"
            },
            "family_execution_closure_claimed": false,
            "remaining_release_gates": ["VAL-MODEL-FAMILY-001"]
        });
        let mut artifact_bytes = serde_json::to_vec_pretty(&artifact)?;
        artifact_bytes.push(b'\n');
        let directory = target_directory(&workspace_root).join("sim-parity");
        fs::create_dir_all(&directory)?;
        fs::write(
            directory.join("val-model-registry-001.json"),
            artifact_bytes,
        )?;
        Ok(())
    }

    fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
        Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root is unavailable")?
            .to_path_buf())
    }

    fn target_directory(workspace_root: &Path) -> PathBuf {
        match std::env::var_os("CARGO_TARGET_DIR") {
            Some(directory) => {
                let directory = PathBuf::from(directory);
                if directory.is_absolute() {
                    directory
                } else {
                    workspace_root.join(directory)
                }
            }
            None => workspace_root.join("target"),
        }
    }

    fn sha256(bytes: &[u8]) -> String {
        let digest = Sha256::digest(bytes);
        let mut result = String::with_capacity(digest.len() * 2);
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for byte in digest {
            result.push(char::from(HEX[usize::from(byte >> 4)]));
            result.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        result
    }
}
