use std::{fs, io, path::PathBuf};

fn refuse<T>(result: io::Result<T>, operation: &str, denied: &mut usize) {
    assert!(result.is_err(), "outside access succeeded: {operation}");
    *denied += 1;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let roots = [PathBuf::from("."), PathBuf::from(std::env::var("PWD")?)];
    let mut denied = 0;
    for root in roots {
        assert_eq!(fs::read(root.join("inside.txt"))?, b"sandbox-positive");
        assert_eq!(
            fs::read(root.join("inside-link/nested.txt"))?,
            b"nested-positive"
        );
        fs::write(root.join("inside-rename.txt"), b"keep-inside")?;
        refuse(
            fs::read(root.join("escape-direct/sensitive.txt")),
            "direct symlink read",
            &mut denied,
        );
        refuse(
            fs::read_dir(root.join("escape-direct/")),
            "trailing slash directory read",
            &mut denied,
        );
        refuse(
            fs::read(root.join("escape-target-slash/sensitive.txt")),
            "target trailing slash read",
            &mut denied,
        );
        refuse(
            fs::read(root.join("escape-file/")),
            "file link trailing slash read",
            &mut denied,
        );
        refuse(
            fs::read(root.join("../outside/sensitive.txt")),
            "parent traversal read",
            &mut denied,
        );
        refuse(
            fs::write(root.join("escape-direct/sensitive.txt"), b"overwritten"),
            "outside overwrite",
            &mut denied,
        );
        refuse(
            fs::write(root.join("escape-target-slash/new.txt"), b"created"),
            "outside create",
            &mut denied,
        );
        refuse(
            fs::create_dir(root.join("escape-direct/new-directory")),
            "outside create directory",
            &mut denied,
        );
        refuse(
            fs::rename(
                root.join("inside-rename.txt"),
                root.join("escape-direct/moved.txt"),
            ),
            "rename out",
            &mut denied,
        );
        refuse(
            fs::rename(
                root.join("escape-direct/sensitive.txt"),
                root.join("moved-from-outside.txt"),
            ),
            "rename in",
            &mut denied,
        );
        refuse(
            fs::remove_file(root.join("escape-target-slash/sensitive.txt")),
            "outside remove file",
            &mut denied,
        );
        refuse(
            fs::remove_dir(root.join("escape-direct/empty")),
            "outside remove directory",
            &mut denied,
        );
        assert_eq!(fs::read(root.join("inside-rename.txt"))?, b"keep-inside");
        fs::rename(
            root.join("inside-rename.txt"),
            root.join("inside-renamed.txt"),
        )?;
        fs::remove_file(root.join("inside-renamed.txt"))?;
        fs::create_dir(root.join("inside-created-directory"))?;
        fs::remove_dir(root.join("inside-created-directory"))?;
    }
    assert_eq!(denied, 24);
    println!("WASI confinement: 24 outside operations denied across both production preopens");
    Ok(())
}
