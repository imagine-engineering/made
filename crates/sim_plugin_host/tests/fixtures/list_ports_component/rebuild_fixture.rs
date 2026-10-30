use std::{error::Error, fs, path::Path};
use wit_component::ComponentEncoder;

const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn main() -> Result<(), Box<dyn Error>> {
    let mut check = false;
    let mut hang = false;
    for argument in std::env::args().skip(1) {
        match argument.as_str() {
            "--check" if !check => check = true,
            "--hang" if !hang => hang = true,
            _ => return Err(format!("unsupported or repeated argument `{argument}`").into()),
        }
    }
    let fixture_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let core_module = if hang {
        fixture_root.join(
            "../hang_component_source/target/wasm32-unknown-unknown/release/sim_plugin_hang_fixture.wasm",
        )
    } else {
        fixture_root.join("target/wasm32-unknown-unknown/release/sim_plugin_echo_fixture.wasm")
    };
    let core_bytes = fs::read(&core_module).map_err(|error| {
        format!(
            "failed to read `{}` ({error}); first run `cargo build --manifest-path {}/Cargo.toml --target wasm32-unknown-unknown --release --lib --offline`",
            core_module.display(),
            fixture_root.display()
        )
    })?;
    let component = ComponentEncoder::default()
        .module(&core_bytes)?
        .validate(true)
        .encode()?;
    assert_no_wasi(&component)?;

    let (destination, output) = if hang {
        (
            fixture_root.join("../hang_component"),
            format!("{}\n", encode_base64(&component)),
        )
    } else {
        let contract = fs::read_to_string(fixture_root.join("port_contract.txt"))?;
        (
            fixture_root.join("../list_ports"),
            format!(
                "{}[component-base64]\n{}\n",
                contract,
                encode_base64(&component)
            ),
        )
    };
    if check {
        if fs::read_to_string(&destination)? != output {
            return Err(format!(
                "compiled fixture `{}` is stale; run the rebuild command without `--check`",
                destination.display()
            )
            .into());
        }
    } else {
        fs::write(destination, output)?;
    }
    Ok(())
}

fn assert_no_wasi(component: &[u8]) -> Result<(), Box<dyn Error>> {
    for forbidden in [b"wasi:".as_slice(), b"wasi_snapshot_preview1".as_slice()] {
        if component
            .windows(forbidden.len())
            .any(|window| window == forbidden)
        {
            return Err("fixture component unexpectedly imports WASI".into());
        }
    }
    Ok(())
}

fn encode_base64(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let Some(&first) = chunk.first() else {
            continue;
        };
        let second = chunk.get(1).copied().unwrap_or(0);
        let third = chunk.get(2).copied().unwrap_or(0);
        encoded.push(char::from(BASE64_ALPHABET[usize::from(first >> 2)]));
        encoded.push(char::from(
            BASE64_ALPHABET[usize::from(((first & 0x03) << 4) | (second >> 4))],
        ));
        encoded.push(if chunk.len() > 1 {
            char::from(BASE64_ALPHABET[usize::from(((second & 0x0f) << 2) | (third >> 6))])
        } else {
            '='
        });
        encoded.push(if chunk.len() > 2 {
            char::from(BASE64_ALPHABET[usize::from(third & 0x3f)])
        } else {
            '='
        });
    }
    encoded
}
