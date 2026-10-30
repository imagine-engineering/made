use std::{env, error::Error, fs};
use wit_component::ComponentEncoder;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let module_path = arguments.next().ok_or("missing core Wasm module path")?;
    let output_path = arguments.next().ok_or("missing component output path")?;
    if arguments.next().is_some() {
        return Err("unexpected fixture rebuild argument".into());
    }
    let module = fs::read(module_path)?;
    let component = ComponentEncoder::default()
        .module(&module)?
        .validate(true)
        .encode()?;
    fs::write(output_path, component)?;
    Ok(())
}
