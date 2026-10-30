pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    #[test]
    fn unit_passes() {
        assert_eq!(super::add(2, 3), 5);
    }

    #[test]
    #[ignore]
    fn ignored_passes() {
        assert_eq!(super::add(1, 1), 2);
    }

    #[test]
    fn cancellable() {
        if std::env::var_os("ZED_RUST_TOOLS_LONG_TEST").is_some() {
            if let Some(marker) = std::env::var_os("ZED_RUST_TOOLS_STARTED_MARKER") {
                std::fs::write(marker, b"started").expect("write cancellation start marker");
            }
            std::thread::sleep(Duration::from_secs(300));
        }
    }
}
