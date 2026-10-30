use std::{collections::BTreeSet, env, fs, io, path::PathBuf};

fn main() -> io::Result<()> {
    // Shared targets may reuse this executable in another checkout; use the invoking Cargo package.
    let manifest_directory = env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "Cargo did not provide the CARGO_MANIFEST_DIR build-script variable",
            )
        })?;
    let mut modules = Vec::new();
    let mut family_modules = Vec::new();
    let mut names = BTreeSet::new();
    let mut descriptor_ids = BTreeSet::new();
    let mut family_descriptor_ids = BTreeSet::new();
    for kind in ["families", "slices"] {
        let directory = manifest_directory.join("src").join(kind);
        let relative_directory = directory
            .strip_prefix(&manifest_directory)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        println!("cargo:rerun-if-changed={}", relative_directory.display());
        if !directory.is_dir() {
            continue;
        }
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            if entry
                .file_name()
                .to_str()
                .is_some_and(|file_name| file_name.starts_with('.'))
            {
                continue;
            }
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
                continue;
            }
            let name = path
                .file_stem()
                .and_then(|value| value.to_str())
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("node module path is not valid UTF-8: {}", path.display()),
                    )
                })?;
            if !valid_module_name(name) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("node module name is invalid: {name}"),
                ));
            }
            if !names.insert(name.to_owned()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("duplicate node module name: {name}"),
                ));
            }
            let relative_path = path
                .strip_prefix(&manifest_directory)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            println!("cargo:rerun-if-changed={}", relative_path.display());
            let source = fs::read_to_string(&path)?;
            let module_descriptor_ids = parse_descriptor_ids(&path, &source)?;
            if kind == "families" && !has_family_bindings_export(&source) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "node family module must declare pub const NATIVE_NODE_BINDINGS: {}",
                        path.display()
                    ),
                ));
            }
            for descriptor_id in module_descriptor_ids {
                if !descriptor_ids.insert(descriptor_id.clone()) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("duplicate node descriptor ID: {descriptor_id}"),
                    ));
                }
                if kind == "families" {
                    family_descriptor_ids.insert(descriptor_id);
                }
            }
            if kind == "families" {
                family_modules.push(name.to_owned());
            }
            modules.push((kind.to_owned(), name.to_owned()));
        }
    }
    modules.sort();
    family_modules.sort();
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
    let descriptor_values = descriptor_ids
        .iter()
        .map(|identifier| format!("{identifier:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    let family_values = family_modules
        .iter()
        .map(|name| format!("\"families/{name}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let family_descriptor_values = family_descriptor_ids
        .iter()
        .map(|identifier| format!("{identifier:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    let family_bindings = if family_modules.is_empty() {
        "Vec::new()".to_owned()
    } else {
        format!(
            "vec![{}].into_iter().flatten().collect::<Vec<_>>()",
            family_modules
                .iter()
                .map(|name| format!("(generated_{name}::NATIVE_NODE_BINDINGS)()?"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let output_directory = env::var_os("OUT_DIR").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "Cargo did not provide the OUT_DIR build-script variable",
        )
    })?;
    fs::write(
        PathBuf::from(output_directory).join("generated_modules.rs"),
        format!(
            "{includes}pub const GENERATED_MODULES: &[&str] = &[{values}];\n\
             pub const GENERATED_DESCRIPTOR_IDS: &[&str] = &[{descriptor_values}];\n\
             pub const GENERATED_FAMILY_MODULES: &[&str] = &[{family_values}];\n\
             pub const GENERATED_FAMILY_DESCRIPTOR_IDS: &[&str] = &[{family_descriptor_values}];\n\
             pub fn generated_family_node_bindings() -> Result<Vec<crate::NativeNodeBinding>, crate::NativeNodeContractError> {{\n\
                 let bindings = {family_bindings};\n\
                 crate::validate_generated_family_bindings(&bindings, GENERATED_FAMILY_DESCRIPTOR_IDS)?;\n\
                 Ok(bindings)\n\
             }}\n"
        ),
    )
}

fn has_family_bindings_export(source: &str) -> bool {
    source.matches("pub const NATIVE_NODE_BINDINGS").count() == 1
}

fn parse_descriptor_ids(path: &std::path::Path, source: &str) -> io::Result<Vec<String>> {
    let declaration = source
        .find("pub const NODE_DESCRIPTOR_IDS")
        .map(|position| &source[position..])
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "node module must declare pub const NODE_DESCRIPTOR_IDS: {}",
                    path.display()
                ),
            )
        })?;
    let value = declaration
        .find('=')
        .map(|position| declaration[position + 1..].trim_start())
        .filter(|value| value.starts_with("&["))
        .ok_or_else(|| invalid_descriptor_declaration(path))?;
    let value = &value[2..];
    let end = value
        .find("];")
        .ok_or_else(|| invalid_descriptor_declaration(path))?;
    let mut remainder = value[..end].trim();
    let mut identifiers = Vec::new();
    while !remainder.is_empty() {
        let quoted = remainder
            .strip_prefix('"')
            .ok_or_else(|| invalid_descriptor_declaration(path))?;
        let quote = quoted
            .find('"')
            .ok_or_else(|| invalid_descriptor_declaration(path))?;
        let identifier = &quoted[..quote];
        if identifier.is_empty()
            || identifier.len() > 4_096
            || identifier.contains('\\')
            || identifier.chars().any(char::is_control)
        {
            return Err(invalid_descriptor_declaration(path));
        }
        identifiers.push(identifier.to_owned());
        remainder = quoted[quote + 1..].trim_start();
        if remainder.is_empty() {
            break;
        }
        remainder = remainder
            .strip_prefix(',')
            .ok_or_else(|| invalid_descriptor_declaration(path))?
            .trim_start();
    }
    if identifiers.is_empty() {
        return Err(invalid_descriptor_declaration(path));
    }
    Ok(identifiers)
}

fn invalid_descriptor_declaration(path: &std::path::Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "node module has an invalid NODE_DESCRIPTOR_IDS declaration: {}",
            path.display()
        ),
    )
}

fn valid_module_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        && name.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
}
