#[path = "operation_contract_evidence.rs"]
mod operation_contract_evidence;

use operation_contract_evidence::{
    ResolutionExpectation, validate_resolution_evidence, validate_resolution_semantics,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs, io,
    path::{Path, PathBuf},
};
use syn::{Expr, ExprArray, ExprStruct, Item, Lit, Member};

const RESOLUTION_FIELDS: &[&str] = &[
    "resolution_module",
    "operation_id",
    "baseline_overload_id",
    "baseline_fixture_sha256",
    "overload_id",
    "ordered_parameters_json",
    "output_arity",
    "output_types_json",
    "rust_signature",
    "mutation_rule",
    "alias_rule",
    "shape_rule",
    "dtype_rule",
    "accumulation_dtype",
    "layout_rule",
    "device_rule",
    "numeric_rule",
    "tolerance",
    "determinism",
    "cancellation_points",
    "vjp_rule",
    "jvp_rule",
    "owner_task_id",
    "evidence_fixture",
    "evidence_fixture_sha256",
];

struct ParsedResolution {
    fields: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BaselineInventoryKind {
    CallableOperation,
    ReclassifiedExternalOperation,
    NamespaceValueReference,
    TypeReference,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BaselineResolutionState {
    ResolvedCallable,
    ResolvedReference,
    ReclassifiedExternalOperation,
    BlockedReceiverUnverified,
    BlockedMissingSemanticsProfile,
    BlockedMissingOracleDependency,
}

impl BaselineResolutionState {
    fn is_blocked(self) -> bool {
        matches!(
            self,
            Self::BlockedReceiverUnverified
                | Self::BlockedMissingSemanticsProfile
                | Self::BlockedMissingOracleDependency
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BaselineContract {
    operation_id: String,
    overload_id: String,
    inventory_kind: BaselineInventoryKind,
    resolution_state: BaselineResolutionState,
    resolution_owner_task_id: String,
    expected_resolution_module: String,
    release_closure_required: bool,
    oracle_fixture_sha256: String,
}

fn main() -> io::Result<()> {
    run_build_validator_self_test()?;
    println!("cargo:rerun-if-changed=src/ops");
    println!("cargo:rerun-if-changed=src/operation_resolutions");
    let mut modules = Vec::new();
    let mut names = BTreeSet::new();
    let directory = PathBuf::from("src/ops");
    if directory.is_dir() {
        for entry in fs::read_dir(&directory)? {
            let path = entry?.path();
            if path.extension().and_then(|value| value.to_str()) != Some("rs") {
                continue;
            }
            let name = module_name(&path)?;
            if !names.insert(name.to_owned()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("duplicate tensor module name: {name}"),
                ));
            }
            modules.push(name.to_owned());
        }
    }
    modules.sort();
    let values = modules
        .iter()
        .map(|name| format!("\"ops/{name}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let includes = modules
        .iter()
        .map(|name| {
            format!(
                "pub mod generated_{name} {{ include!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/src/ops/{name}.rs\")); }}\n"
            )
        })
        .collect::<String>();
    let output = PathBuf::from(env::var_os("OUT_DIR").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "Cargo did not provide the OUT_DIR build-script variable",
        )
    })?);
    fs::write(
        output.join("generated_modules.rs"),
        format!("{includes}pub const GENERATED_MODULES: &[&str] = &[{values}];\n"),
    )?;
    write_operation_resolutions(&output, &names)
}

fn write_operation_resolutions(
    output: &std::path::Path,
    operation_modules: &BTreeSet<String>,
) -> io::Result<()> {
    println!("cargo:rerun-if-changed=src/operation_contract_records.rs");
    println!("cargo:rerun-if-changed=operation_contract_evidence.rs");
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| io::Error::other("sim_tensor is not below the workspace crates directory"))?
        .to_path_buf();
    let baseline_contracts =
        read_baseline_contracts(Path::new("src/operation_contract_records.rs"))?;
    let resolution_directory = PathBuf::from("src/operation_resolutions");
    let mut resolution_modules = Vec::new();
    let mut resolution_names = BTreeSet::new();
    let mut parsed_by_module = BTreeMap::new();
    let mut sealed_resolutions = Vec::new();
    let mut resolved_operation_ids = BTreeSet::new();
    let mut resolved_overload_ids = BTreeSet::new();
    if resolution_directory.is_dir() {
        for entry in fs::read_dir(&resolution_directory)? {
            let path = entry?.path();
            if path.extension().and_then(|value| value.to_str()) != Some("rs") {
                continue;
            }
            let name = module_name(&path)?;
            if !resolution_names.insert(name.to_owned()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("duplicate tensor operation-resolution module name: {name}"),
                ));
            }
            if !operation_modules.contains(name) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "tensor operation-resolution module has no paired operation module: {}",
                        path.display()
                    ),
                ));
            }
            println!("cargo:rerun-if-changed={}", path.display());
            let parsed = read_resolution_source(&path)?;
            for resolution in &parsed {
                record_unique_resolution(
                    resolution,
                    &mut resolved_operation_ids,
                    &mut resolved_overload_ids,
                )?;
                validate_build_resolution(&workspace_root, name, resolution, &baseline_contracts)?;
                let evidence_fixture = resolution.field("evidence_fixture")?;
                let evidence_path = workspace_root.join(evidence_fixture);
                let evidence_module = evidence_path.parent().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("resolution evidence has no module parent: {evidence_fixture}"),
                    )
                })?;
                println!("cargo:rerun-if-changed={}", evidence_module.display());
                println!("cargo:rerun-if-changed={}", evidence_path.display());
                sealed_resolutions.push((
                    resolution.field("operation_id")?.to_owned(),
                    resolution.field("overload_id")?.to_owned(),
                    name.to_owned(),
                    resolution.field("evidence_fixture_sha256")?.to_owned(),
                ));
            }
            parsed_by_module.insert(name.to_owned(), parsed);
            resolution_modules.push(name.to_owned());
        }
    }
    resolution_modules.sort();
    let includes = resolution_modules
        .iter()
        .map(|name| {
            let records = parsed_by_module
                .get(name)
                .expect("every discovered module has parsed resolution records")
                .iter()
                .map(ParsedResolution::render_sealed)
                .collect::<Result<Vec<_>, _>>()?
                .join(",\n");
            Ok::<_, io::Error>(format!(
                "pub mod generated_resolution_{name} {{\n    use super::{{OperationResolutionBuildSeal, ResolvedOperationContract}};\n    pub static CONTRACTS: &[ResolvedOperationContract] = &[{records}];\n}}\n"
            ))
        })
        .collect::<Result<Vec<_>, _>>()?
        .join("");
    let slices = resolution_modules
        .iter()
        .map(|name| {
            format!(
                "GeneratedOperationResolutionSlice {{ module_name: \"{name}\", contracts: generated_resolution_{name}::CONTRACTS }}"
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let modules = resolution_modules
        .iter()
        .map(|name| format!("\"operation_resolutions/{name}\""))
        .collect::<Vec<_>>()
        .join(", ");
    sealed_resolutions.sort();
    let sealed = sealed_resolutions
        .iter()
        .map(|(operation_id, overload_id, module_name, digest)| {
            format!("({operation_id:?}, {overload_id:?}, {module_name:?}, {digest:?})")
        })
        .collect::<Vec<_>>()
        .join(", ");
    fs::write(
        output.join("generated_operation_resolutions.rs"),
        format!(
            "{includes}pub static GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES: &[GeneratedOperationResolutionSlice] = &[{slices}];\npub static GENERATED_BUILD_SEALED_OPERATION_RESOLUTIONS: &[(&str, &str, &str, &str)] = &[{sealed}];\npub const GENERATED_OPERATION_RESOLUTION_MODULES: &[&str] = &[{modules}];\npub const GENERATED_OPERATION_RESOLUTION_SOURCE_DIRECTORY: &str = \"src/operation_resolutions\";\n"
        ),
    )
}

impl ParsedResolution {
    fn field(&self, name: &str) -> io::Result<&str> {
        self.fields.get(name).map(String::as_str).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("operation resolution is missing string field {name}"),
            )
        })
    }

    fn expectation(&self) -> io::Result<ResolutionExpectation<'_>> {
        Ok(ResolutionExpectation {
            resolution_module: self.field("resolution_module")?,
            operation_id: self.field("operation_id")?,
            baseline_overload_id: self.field("baseline_overload_id")?,
            baseline_fixture_sha256: self.field("baseline_fixture_sha256")?,
            overload_id: self.field("overload_id")?,
            ordered_parameters_json: self.field("ordered_parameters_json")?,
            output_arity: self.field("output_arity")?,
            output_types_json: self.field("output_types_json")?,
            rust_signature: self.field("rust_signature")?,
            mutation_rule: self.field("mutation_rule")?,
            alias_rule: self.field("alias_rule")?,
            shape_rule: self.field("shape_rule")?,
            dtype_rule: self.field("dtype_rule")?,
            accumulation_dtype: self.field("accumulation_dtype")?,
            layout_rule: self.field("layout_rule")?,
            device_rule: self.field("device_rule")?,
            numeric_rule: self.field("numeric_rule")?,
            tolerance: self.field("tolerance")?,
            determinism: self.field("determinism")?,
            cancellation_points: self.field("cancellation_points")?,
            vjp_rule: self.field("vjp_rule")?,
            jvp_rule: self.field("jvp_rule")?,
            owner_task_id: self.field("owner_task_id")?,
            evidence_fixture: self.field("evidence_fixture")?,
            evidence_fixture_sha256: self.field("evidence_fixture_sha256")?,
        })
    }

    fn render_sealed(&self) -> io::Result<String> {
        let fields = RESOLUTION_FIELDS
            .iter()
            .map(|field| Ok(format!("{field}: {:?}", self.field(field)?)))
            .collect::<io::Result<Vec<_>>>()?
            .join(", ");
        Ok(format!(
            "ResolvedOperationContract {{ {fields}, build_seal: OperationResolutionBuildSeal }}"
        ))
    }
}

fn read_resolution_source(path: &Path) -> io::Result<Vec<ParsedResolution>> {
    let source = fs::read_to_string(path)?;
    let expression = syn::parse_str::<Expr>(&source).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("operation resolution source is not one expression: {error}"),
        )
    })?;
    parse_resolution_array(expression).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid operation resolution source {}: {error}",
                path.display()
            ),
        )
    })
}

fn parse_resolution_array(expression: Expr) -> Result<Vec<ParsedResolution>, String> {
    let expression = match expression {
        Expr::Reference(reference) => *reference.expr,
        expression => expression,
    };
    let Expr::Array(ExprArray { elems, .. }) = expression else {
        return Err("source must be an array or reference to an array".to_owned());
    };
    elems
        .into_iter()
        .map(|expression| {
            let Expr::Struct(expression) = expression else {
                return Err(
                    "every resolution must be a ResolvedOperationContract literal".to_owned(),
                );
            };
            parse_resolution_struct(expression)
        })
        .collect()
}

fn parse_resolution_struct(expression: ExprStruct) -> Result<ParsedResolution, String> {
    if expression.rest.is_some()
        || expression
            .path
            .segments
            .last()
            .is_none_or(|segment| segment.ident != "ResolvedOperationContract")
    {
        return Err("resolution must be a complete ResolvedOperationContract literal".to_owned());
    }
    let mut fields = BTreeMap::new();
    for field in expression.fields {
        let Member::Named(name) = field.member else {
            return Err("resolution fields must be named".to_owned());
        };
        let Expr::Lit(literal) = field.expr else {
            return Err(format!("resolution field {name} must be a string literal"));
        };
        let Lit::Str(value) = literal.lit else {
            return Err(format!("resolution field {name} must be a string literal"));
        };
        if fields.insert(name.to_string(), value.value()).is_some() {
            return Err(format!("resolution field {name} is duplicated"));
        }
    }
    let actual_fields = fields.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected_fields = RESOLUTION_FIELDS.iter().copied().collect::<BTreeSet<_>>();
    if actual_fields != expected_fields {
        return Err("resolution fields do not match the sealed schema".to_owned());
    }
    Ok(ParsedResolution { fields })
}

fn read_baseline_contracts(path: &Path) -> io::Result<BTreeMap<String, BaselineContract>> {
    let source = fs::read_to_string(path)?;
    parse_baseline_contracts(&source)
}

fn parse_baseline_contracts(source: &str) -> io::Result<BTreeMap<String, BaselineContract>> {
    let syntax = syn::parse_file(source).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("generated operation contract table is invalid: {error}"),
        )
    })?;
    let table = syntax.items.into_iter().find_map(|item| match item {
        Item::Static(item) if item.ident == "OPERATION_CONTRACTS" => Some(item.expr),
        _ => None,
    });
    let Some(expression) = table else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "generated operation contract table has no OPERATION_CONTRACTS static",
        ));
    };
    let expression = match *expression {
        Expr::Reference(reference) => *reference.expr,
        expression => expression,
    };
    let Expr::Array(array) = expression else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "generated operation contract table is not an array",
        ));
    };
    let mut contracts = BTreeMap::new();
    for expression in array.elems {
        let Expr::Struct(record) = expression else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "generated operation contract table contains a non-record",
            ));
        };
        if record.rest.is_some()
            || record
                .path
                .segments
                .last()
                .is_none_or(|segment| segment.ident != "OperationContractRecord")
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "baseline record must be a complete OperationContractRecord literal",
            ));
        }
        let mut fields = BTreeMap::new();
        for field in record.fields {
            let Member::Named(name) = field.member else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "baseline record fields must be named",
                ));
            };
            if fields.insert(name.to_string(), field.expr).is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("baseline field {name} is duplicated"),
                ));
            }
        }
        let contract = BaselineContract {
            operation_id: baseline_string_field(&fields, "operation_id")?,
            overload_id: baseline_string_field(&fields, "overload_id")?,
            inventory_kind: parse_baseline_inventory_kind(baseline_field(
                &fields,
                "inventory_kind",
            )?)?,
            resolution_state: parse_baseline_resolution_state(baseline_field(
                &fields,
                "resolution_state",
            )?)?,
            resolution_owner_task_id: baseline_string_field(&fields, "resolution_owner_task_id")?,
            expected_resolution_module: baseline_string_field(
                &fields,
                "expected_resolution_module",
            )?,
            release_closure_required: baseline_bool_field(&fields, "release_closure_required")?,
            oracle_fixture_sha256: baseline_string_field(&fields, "oracle_fixture_sha256")?,
        };
        if contracts
            .insert(contract.operation_id.clone(), contract.clone())
            .is_some()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("duplicate baseline operation ID: {}", contract.operation_id),
            ));
        }
    }
    Ok(contracts)
}

fn baseline_field<'a>(fields: &'a BTreeMap<String, Expr>, name: &str) -> io::Result<&'a Expr> {
    fields.get(name).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("baseline record has no {name} field"),
        )
    })
}

fn baseline_string_field(fields: &BTreeMap<String, Expr>, name: &str) -> io::Result<String> {
    let Expr::Lit(literal) = baseline_field(fields, name)? else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("baseline field {name} must be a string literal"),
        ));
    };
    let Lit::Str(value) = &literal.lit else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("baseline field {name} must be a string literal"),
        ));
    };
    Ok(value.value())
}

fn baseline_bool_field(fields: &BTreeMap<String, Expr>, name: &str) -> io::Result<bool> {
    let Expr::Lit(literal) = baseline_field(fields, name)? else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("baseline field {name} must be a boolean literal"),
        ));
    };
    let Lit::Bool(value) = &literal.lit else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("baseline field {name} must be a boolean literal"),
        ));
    };
    Ok(value.value)
}

fn baseline_variant(expression: &Expr, name: &str) -> io::Result<String> {
    let Expr::Path(path) = expression else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("baseline field {name} must be an enum variant"),
        ));
    };
    if path.qself.is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("baseline field {name} must be an enum variant"),
        ));
    }
    path.path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("baseline field {name} has no enum variant"),
            )
        })
}

fn parse_baseline_inventory_kind(expression: &Expr) -> io::Result<BaselineInventoryKind> {
    let variant = baseline_variant(expression, "inventory_kind")?;
    match variant.as_str() {
        "CallableOperation" => Ok(BaselineInventoryKind::CallableOperation),
        "ReclassifiedExternalOperation" => Ok(BaselineInventoryKind::ReclassifiedExternalOperation),
        "NamespaceValueReference" => Ok(BaselineInventoryKind::NamespaceValueReference),
        "TypeReference" => Ok(BaselineInventoryKind::TypeReference),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown baseline inventory kind: {variant}"),
        )),
    }
}

fn parse_baseline_resolution_state(expression: &Expr) -> io::Result<BaselineResolutionState> {
    let variant = baseline_variant(expression, "resolution_state")?;
    match variant.as_str() {
        "ResolvedCallable" => Ok(BaselineResolutionState::ResolvedCallable),
        "ResolvedReference" => Ok(BaselineResolutionState::ResolvedReference),
        "ReclassifiedExternalOperation" => {
            Ok(BaselineResolutionState::ReclassifiedExternalOperation)
        }
        "BlockedReceiverUnverified" => Ok(BaselineResolutionState::BlockedReceiverUnverified),
        "BlockedMissingSemanticsProfile" => {
            Ok(BaselineResolutionState::BlockedMissingSemanticsProfile)
        }
        "BlockedMissingOracleDependency" => {
            Ok(BaselineResolutionState::BlockedMissingOracleDependency)
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown baseline resolution state: {variant}"),
        )),
    }
}

fn validate_build_resolution(
    workspace_root: &Path,
    module_name: &str,
    resolution: &ParsedResolution,
    baseline_contracts: &BTreeMap<String, BaselineContract>,
) -> io::Result<()> {
    let expectation = resolution.expectation()?;
    let Some(baseline) = baseline_contracts.get(expectation.operation_id) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "resolution has no generated baseline: {}",
                expectation.operation_id
            ),
        ));
    };
    validate_baseline_transition(module_name, &expectation, baseline)?;
    validate_resolution_semantics(&expectation).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("resolution semantics failed build validation: {error}"),
        )
    })?;
    validate_resolution_evidence(workspace_root, &expectation).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("resolution evidence failed build validation: {error}"),
        )
    })
}

fn validate_baseline_transition(
    module_name: &str,
    expectation: &ResolutionExpectation<'_>,
    baseline: &BaselineContract,
) -> io::Result<()> {
    if baseline.inventory_kind != BaselineInventoryKind::CallableOperation
        || !baseline.resolution_state.is_blocked()
        || !baseline.release_closure_required
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "resolution baseline is not a blocked release-closing callable operation: {}",
                expectation.operation_id
            ),
        ));
    }
    if expectation.operation_id != baseline.operation_id
        || expectation.baseline_overload_id != baseline.overload_id
        || expectation.baseline_fixture_sha256 != baseline.oracle_fixture_sha256
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "resolution baseline identity does not match generated discovery: {}",
                expectation.operation_id
            ),
        ));
    }
    if expectation.overload_id == baseline.overload_id
        || expectation
            .overload_id
            .to_ascii_lowercase()
            .contains("blocked")
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "resolved overload does not replace its blocked baseline: {}",
                expectation.operation_id
            ),
        ));
    }
    if expectation.resolution_module != module_name
        || baseline.expected_resolution_module != module_name
        || baseline.resolution_owner_task_id != expectation.owner_task_id
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "resolution owner/module does not match its generated baseline: {}",
                expectation.operation_id
            ),
        ));
    }
    Ok(())
}

fn record_unique_resolution(
    resolution: &ParsedResolution,
    operation_ids: &mut BTreeSet<String>,
    overload_ids: &mut BTreeSet<String>,
) -> io::Result<()> {
    let operation_id = resolution.field("operation_id")?;
    if !operation_ids.insert(operation_id.to_owned()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("duplicate compiled resolution operation ID: {operation_id}"),
        ));
    }
    let overload_id = resolution.field("overload_id")?;
    if !overload_ids.insert(overload_id.to_owned()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("duplicate compiled resolution overload ID: {overload_id}"),
        ));
    }
    Ok(())
}

fn run_build_validator_self_test() -> io::Result<()> {
    const BASELINE_DIGEST: &str =
        "1111111111111111111111111111111111111111111111111111111111111111";
    let source = format!(
        r#"
pub static OPERATION_CONTRACTS: &[OperationContractRecord] = &[
    OperationContractRecord {{
        operation_id: "operation",
        overload_id: "operation:blocked",
        inventory_kind: ContractInventoryKind::CallableOperation,
        resolution_state: ContractResolutionState::BlockedMissingSemanticsProfile,
        resolution_owner_task_id: "owner",
        expected_resolution_module: "module",
        release_closure_required: true,
        oracle_fixture_sha256: "{BASELINE_DIGEST}",
    }},
];
"#
    );
    let contracts = parse_baseline_contracts(&source)?;
    let baseline = contracts.get("operation").ok_or_else(|| {
        io::Error::other("build-validator self-test did not parse its baseline operation")
    })?;
    let expectation = || ResolutionExpectation {
        resolution_module: "module",
        operation_id: "operation",
        baseline_overload_id: "operation:blocked",
        baseline_fixture_sha256: BASELINE_DIGEST,
        overload_id: "operation:resolved-v1",
        ordered_parameters_json: "[]",
        output_arity: "1",
        output_types_json: "[]",
        rust_signature: "signature",
        mutation_rule: "none",
        alias_rule: "none",
        shape_rule: "scalar",
        dtype_rule: "f32",
        accumulation_dtype: "f32",
        layout_rule: "contiguous",
        device_rule: "cpu",
        numeric_rule: "exact",
        tolerance: "exact",
        determinism: "deterministic",
        cancellation_points: "entry",
        vjp_rule: "forward-only",
        jvp_rule: "forward-only",
        owner_task_id: "owner",
        evidence_fixture: "unused.json",
        evidence_fixture_sha256: BASELINE_DIGEST,
    };
    validate_baseline_transition("module", &expectation(), baseline).map_err(|error| {
        io::Error::other(format!(
            "build-validator self-test rejected a valid baseline transition: {error}"
        ))
    })?;

    let mut mutated_baseline = baseline.clone();
    mutated_baseline.inventory_kind = BaselineInventoryKind::NamespaceValueReference;
    expect_self_test_rejection(
        validate_baseline_transition("module", &expectation(), &mutated_baseline),
        "namespace reference",
    )?;
    mutated_baseline.inventory_kind = BaselineInventoryKind::TypeReference;
    expect_self_test_rejection(
        validate_baseline_transition("module", &expectation(), &mutated_baseline),
        "type reference",
    )?;
    mutated_baseline.inventory_kind = BaselineInventoryKind::ReclassifiedExternalOperation;
    expect_self_test_rejection(
        validate_baseline_transition("module", &expectation(), &mutated_baseline),
        "reclassified external operation",
    )?;
    mutated_baseline.inventory_kind = BaselineInventoryKind::CallableOperation;
    mutated_baseline.resolution_state = BaselineResolutionState::ResolvedReference;
    expect_self_test_rejection(
        validate_baseline_transition("module", &expectation(), &mutated_baseline),
        "non-blocked resolution state",
    )?;
    mutated_baseline.resolution_state = BaselineResolutionState::BlockedMissingSemanticsProfile;
    mutated_baseline.release_closure_required = false;
    expect_self_test_rejection(
        validate_baseline_transition("module", &expectation(), &mutated_baseline),
        "non-release-closing baseline",
    )?;

    let wrong_overload = ResolutionExpectation {
        baseline_overload_id: "operation:other-blocked",
        ..expectation()
    };
    expect_self_test_rejection(
        validate_baseline_transition("module", &wrong_overload, baseline),
        "mismatched baseline overload",
    )?;
    let wrong_digest = ResolutionExpectation {
        baseline_fixture_sha256: "2222222222222222222222222222222222222222222222222222222222222222",
        ..expectation()
    };
    expect_self_test_rejection(
        validate_baseline_transition("module", &wrong_digest, baseline),
        "mismatched baseline fixture digest",
    )?;
    let blocked_overload = ResolutionExpectation {
        overload_id: "operation:blocked",
        ..expectation()
    };
    expect_self_test_rejection(
        validate_baseline_transition("module", &blocked_overload, baseline),
        "unchanged blocked overload",
    )?;
    let wrong_owner = ResolutionExpectation {
        owner_task_id: "other-owner",
        ..expectation()
    };
    expect_self_test_rejection(
        validate_baseline_transition("module", &wrong_owner, baseline),
        "cross-owner resolution",
    )?;
    expect_self_test_rejection(
        validate_baseline_transition("other_module", &expectation(), baseline),
        "cross-module resolution",
    )?;

    let resolution = ParsedResolution {
        fields: BTreeMap::from([
            ("operation_id".to_owned(), "operation".to_owned()),
            ("overload_id".to_owned(), "operation:resolved-v1".to_owned()),
        ]),
    };
    let duplicate_operation = ParsedResolution {
        fields: BTreeMap::from([
            ("operation_id".to_owned(), "operation".to_owned()),
            ("overload_id".to_owned(), "operation:resolved-v2".to_owned()),
        ]),
    };
    let duplicate_overload = ParsedResolution {
        fields: BTreeMap::from([
            ("operation_id".to_owned(), "other-operation".to_owned()),
            ("overload_id".to_owned(), "operation:resolved-v1".to_owned()),
        ]),
    };
    let mut operation_ids = BTreeSet::new();
    let mut overload_ids = BTreeSet::new();
    record_unique_resolution(&resolution, &mut operation_ids, &mut overload_ids)?;
    expect_self_test_rejection(
        record_unique_resolution(&duplicate_operation, &mut operation_ids, &mut overload_ids),
        "duplicate operation ID",
    )?;
    let mut operation_ids = BTreeSet::from(["operation".to_owned()]);
    let mut overload_ids = BTreeSet::from(["operation:resolved-v1".to_owned()]);
    expect_self_test_rejection(
        record_unique_resolution(&duplicate_overload, &mut operation_ids, &mut overload_ids),
        "duplicate overload ID",
    )
}

fn expect_self_test_rejection(result: io::Result<()>, mutation: &str) -> io::Result<()> {
    if result.is_ok() {
        return Err(io::Error::other(format!(
            "build-validator self-test accepted {mutation}"
        )));
    }
    Ok(())
}

fn module_name(path: &std::path::Path) -> io::Result<&str> {
    let name = path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor module path is not valid UTF-8: {}", path.display()),
            )
        })?;
    if !valid_module_name(name) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tensor module name is invalid: {name}"),
        ));
    }
    Ok(name)
}

fn valid_module_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        && name.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
}
