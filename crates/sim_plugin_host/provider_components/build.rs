use std::{
    collections::BTreeSet,
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

const MAXIMUM_PROVIDERS: usize = 256;

fn main() -> Result<(), Box<dyn Error>> {
    let manifest_directory = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR")
            .ok_or("CARGO_MANIFEST_DIR is unavailable for provider component discovery")?,
    );
    let vendor_directory = manifest_directory.join("src/vendors");
    let fixture_directory = manifest_directory.join("../../sim_test_support/fixtures/providers");
    println!("cargo:rerun-if-changed={}", vendor_directory.display());
    println!("cargo:rerun-if-changed={}", fixture_directory.display());

    let vendors = discover_rust_modules(&vendor_directory)?;
    let fixtures = discover_fixture_directories(&fixture_directory)?;
    if vendors != fixtures {
        return Err(format!(
            "provider modules and fixtures differ: modules={vendors:?}, fixtures={fixtures:?}"
        )
        .into());
    }
    if vendors.len() > MAXIMUM_PROVIDERS {
        return Err("provider component count exceeds the reviewed bound".into());
    }

    let mut generated = String::from("pub mod vendors {\n");
    for name in &vendors {
        generated.push_str("pub mod ");
        generated.push_str(name);
        generated.push_str(" { include!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/src/vendors/");
        generated.push_str(name);
        generated.push_str(".rs\")); }\n");
    }
    generated.push_str("}\n");
    generated.push_str(
        "pub static DISCOVERED_PROVIDER_COMPONENTS: &[DiscoveredProviderComponent] = &[\n",
    );
    for name in &vendors {
        generated.push_str("DiscoveredProviderComponent { name: \"");
        generated.push_str(name);
        generated.push_str("\", fixture: include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/../../sim_test_support/fixtures/providers/");
        generated.push_str(name);
        generated.push_str("/conformance.fixture\")), conformance: vendors::");
        generated.push_str(name);
        generated.push_str("::run_conformance },\n");
    }
    generated.push_str("];\n");

    let output_directory = PathBuf::from(
        env::var_os("OUT_DIR").ok_or("OUT_DIR is unavailable for provider component discovery")?,
    );
    fs::write(output_directory.join("provider_registry.rs"), generated)?;
    Ok(())
}

fn discover_rust_modules(directory: &Path) -> Result<BTreeSet<String>, Box<dyn Error>> {
    if !directory.exists() {
        return Ok(BTreeSet::new());
    }
    let mut names = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() || !file_type.is_file() {
            return Err(format!(
                "provider module is not a regular file: {}",
                entry.path().display()
            )
            .into());
        }
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("rs") {
            return Err(format!(
                "provider module has a non-Rust extension: {}",
                path.display()
            )
            .into());
        }
        let name = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or("provider module name is not UTF-8")?;
        validate_name(name)?;
        names.insert(name.to_owned());
    }
    Ok(names)
}

fn discover_fixture_directories(directory: &Path) -> Result<BTreeSet<String>, Box<dyn Error>> {
    let mut names = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| "provider fixture name is not UTF-8")?;
        if name == "common" {
            continue;
        }
        let file_type = entry.file_type()?;
        if file_type.is_symlink() || !file_type.is_dir() {
            return Err(format!(
                "provider fixture is not a regular directory: {}",
                entry.path().display()
            )
            .into());
        }
        validate_name(&name)?;
        let conformance = entry.path().join("conformance.fixture");
        if !conformance.is_file() {
            return Err(format!(
                "provider fixture lacks conformance.fixture: {}",
                entry.path().display()
            )
            .into());
        }
        names.insert(name);
    }
    Ok(names)
}

fn validate_name(name: &str) -> Result<(), Box<dyn Error>> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        || name.starts_with(|character: char| character.is_ascii_digit())
    {
        return Err(format!("invalid provider component name: {name}").into());
    }
    Ok(())
}
