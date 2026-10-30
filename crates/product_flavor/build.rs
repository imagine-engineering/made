use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-env-changed=ZED_PRODUCT_ID");
    println!("cargo:rerun-if-env-changed=ZED_PRODUCT_DEV_PREVIEW");
    println!("cargo:rerun-if-env-changed=ZED_RELEASE_CHANNEL");
    println!("cargo:rerun-if-changed=../zed/RELEASE_CHANNEL");
    println!("cargo:rustc-check-cfg=cfg(product_flavor_rust)");
    println!("cargo:rustc-check-cfg=cfg(product_flavor_jvm)");
    println!(
        "cargo:rustc-check-cfg=cfg(product_release_channel, values(\"dev\", \"nightly\", \"preview\", \"stable\"))"
    );

    let release_channel = env::var("ZED_RELEASE_CHANNEL").unwrap_or_else(|_| {
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../zed/RELEASE_CHANNEL"))
            .unwrap_or_else(|_| "stable".to_string())
    });
    let release_channel = release_channel.trim();
    match release_channel {
        "dev" | "nightly" | "preview" | "stable" => {
            println!("cargo:rustc-cfg=product_release_channel=\"{release_channel}\"")
        }
        _ => return Err(format!("unknown release channel `{release_channel}`").into()),
    }

    let product_id = env::var("ZED_PRODUCT_ID").unwrap_or_else(|_| "rust".to_string());
    match product_id.as_str() {
        "rust" => println!("cargo:rustc-cfg=product_flavor_rust"),
        "jvm"
            if env::var("ZED_PRODUCT_DEV_PREVIEW").as_deref() == Ok("1")
                && env::var("PROFILE").as_deref() == Ok("debug")
                && release_channel == "dev" =>
        {
            println!("cargo:rustc-cfg=product_flavor_jvm");
        }
        "jvm" | "game" => {
            return Err(format!("product `{product_id}` is planned and cannot be built").into());
        }
        _ => return Err(format!("unknown product `{product_id}`").into()),
    }

    Ok(())
}
