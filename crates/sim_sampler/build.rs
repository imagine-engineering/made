use std::{
    collections::BTreeSet,
    env, fs, io,
    path::{Path, PathBuf},
};

fn main() -> io::Result<()> {
    let mut modules = Vec::new();
    let mut names = BTreeSet::new();
    for kind in ["algorithms", "schedulers"] {
        let directory = PathBuf::from("src").join(kind);
        println!("cargo:rerun-if-changed={}", directory.display());
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
                        format!("duplicate sampler module name: {name}"),
                    ));
                }
                modules.push((kind.to_owned(), name.to_owned(), path));
            }
        }
    }
    modules.sort();
    let values = modules
        .iter()
        .map(|(kind, name, _)| format!("\"{kind}/{name}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let includes = modules
        .iter()
        .map(|(kind, name, _)| {
            format!(
                "pub mod generated_{name} {{ include!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/src/{kind}/{name}.rs\")); }}\n"
            )
        })
        .collect::<String>();
    let row_modules = modules
        .iter()
        .filter(|(_, name, _)| is_row_module(name))
        .collect::<Vec<_>>();
    validate_row_closure(&row_modules)?;
    let sampler_definitions = definition_references(&row_modules, "algorithms");
    let scheduler_definitions = definition_references(&row_modules, "schedulers");
    let output_directory = env::var_os("OUT_DIR").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "Cargo did not provide the OUT_DIR build-script variable",
        )
    })?;
    fs::write(
        PathBuf::from(&output_directory).join("generated_modules.rs"),
        format!(
            "{includes}pub const GENERATED_MODULES: &[&str] = &[{values}];\n\
             pub const GENERATED_SAMPLER_DEFINITIONS: &[crate::sampler::SamplerDefinition] = &[{sampler_definitions}];\n\
             pub const GENERATED_SCHEDULER_DEFINITIONS: &[crate::scheduler::SchedulerDefinition] = &[{scheduler_definitions}];\n"
        ),
    )?;
    write_test_harness(
        &PathBuf::from(&output_directory),
        "algorithms",
        "generated_sampler_tests.rs",
        &row_modules,
    )?;
    write_test_harness(
        &PathBuf::from(output_directory),
        "schedulers",
        "generated_scheduler_tests.rs",
        &row_modules,
    )
}

fn is_row_module(name: &str) -> bool {
    name.rsplit_once("_sim_model_")
        .is_some_and(|(prefix, suffix)| {
            !prefix.is_empty()
                && suffix.len() == 4
                && suffix.bytes().all(|byte| byte.is_ascii_digit())
        })
}

fn definition_references(modules: &[&(String, String, PathBuf)], kind: &str) -> String {
    modules
        .iter()
        .filter(|(module_kind, _, _)| module_kind == kind)
        .map(|(_, name, _)| format!("generated_{name}::DEFINITION"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn validate_row_closure(modules: &[&(String, String, PathBuf)]) -> io::Result<()> {
    let mut expected = BTreeSet::new();
    for (kind, name, _) in modules {
        let fixture_kind = match kind.as_str() {
            "algorithms" => "samplers",
            "schedulers" => "schedulers",
            _ => continue,
        };
        let test = PathBuf::from("tests").join(kind).join(format!("{name}.rs"));
        let fixture = PathBuf::from("../sim_test_support/fixtures")
            .join(fixture_kind)
            .join(name.replace("_sim_model_", "_comfy_model_"));
        println!("cargo:rerun-if-changed={}", test.display());
        println!("cargo:rerun-if-changed={}", fixture.display());
        if !test.is_file() || !fixture.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "sampler row {kind}/{name} requires test {} and fixture {}",
                    test.display(),
                    fixture.display()
                ),
            ));
        }
        if !expected.insert((kind.clone(), name.clone())) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("duplicate generated sampler row {kind}/{name}"),
            ));
        }
    }
    for kind in ["algorithms", "schedulers"] {
        let test_directory = PathBuf::from("tests").join(kind);
        println!("cargo:rerun-if-changed={}", test_directory.display());
        if test_directory.is_dir() {
            for entry in fs::read_dir(&test_directory)? {
                let path = entry?.path();
                if path.extension().and_then(|value| value.to_str()) != Some("rs") {
                    continue;
                }
                let name = module_name(&path)?;
                if is_row_module(name) && !expected.contains(&(kind.to_owned(), name.to_owned())) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("orphan generated sampler test {}", path.display()),
                    ));
                }
            }
        }
        let fixture_kind = if kind == "algorithms" {
            "samplers"
        } else {
            "schedulers"
        };
        let fixture_directory = PathBuf::from("../sim_test_support/fixtures").join(fixture_kind);
        println!("cargo:rerun-if-changed={}", fixture_directory.display());
        if fixture_directory.is_dir() {
            for entry in fs::read_dir(&fixture_directory)? {
                let entry = entry?;
                if !entry.file_type()?.is_dir() {
                    continue;
                }
                let name = entry.file_name().into_string().map_err(|value| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("sampler fixture name is not valid UTF-8: {value:?}"),
                    )
                })?;
                if is_row_module(&name) && !expected.contains(&(kind.to_owned(), name.clone())) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("orphan generated sampler fixture {fixture_kind}/{name}"),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn write_test_harness(
    output_directory: &Path,
    kind: &str,
    filename: &str,
    modules: &[&(String, String, PathBuf)],
) -> io::Result<()> {
    let body = modules
        .iter()
        .filter(|(module_kind, _, _)| module_kind == kind)
        .map(|(_, name, _)| {
            format!(
                "mod generated_{name} {{ include!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/tests/{kind}/{name}.rs\")); }}\n"
            )
        })
        .collect::<String>();
    fs::write(output_directory.join(filename), body)
}

fn module_name(path: &std::path::Path) -> io::Result<&str> {
    let name = path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("sampler module path is not valid UTF-8: {}", path.display()),
            )
        })?;
    if !valid_module_name(name) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("sampler module name is invalid: {name}"),
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
