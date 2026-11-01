use serde_json::json;
use settings::{MergeFromTrait as _, RootUserSettings as _, Settings, SettingsContent};
use std::path::Path;
use util::rel_path::rel_path;
use util::{
    paths::{PathMatcher, PathStyle},
    rel_path::RelPath,
};
use worktree::*;

fn make_settings_with_read_only(patterns: &[&str]) -> WorktreeSettings {
    WorktreeSettings {
        prevent_sharing_in_public_channels: false,
        file_scan_exclusions: PathMatcher::default(),
        file_scan_inclusions: PathMatcher::default(),
        parent_dir_scan_inclusions: PathMatcher::default(),
        private_files: PathMatcher::default(),
        hidden_files: PathMatcher::default(),
        read_only_files: PathMatcher::new(
            patterns.iter().map(|s| s.to_string()),
            PathStyle::local(),
        )
        .unwrap(),
        scan_symlinks: Default::default(),
    }
}

#[test]
fn test_is_path_read_only_with_glob_patterns() {
    let settings = make_settings_with_read_only(&["**/generated/**", "**/*.gen.rs"]);

    let generated_file =
        RelPath::new(Path::new("src/generated/schema.rs"), PathStyle::local()).unwrap();
    assert!(
        settings.is_path_read_only(&generated_file),
        "Files in generated directory should be read-only"
    );

    let gen_rs_file = RelPath::new(Path::new("src/types.gen.rs"), PathStyle::local()).unwrap();
    assert!(
        settings.is_path_read_only(&gen_rs_file),
        "Files with .gen.rs extension should be read-only"
    );

    let regular_file = RelPath::new(Path::new("src/main.rs"), PathStyle::local()).unwrap();
    assert!(
        !settings.is_path_read_only(&regular_file),
        "Regular files should not be read-only"
    );

    let similar_name = RelPath::new(Path::new("src/generator.rs"), PathStyle::local()).unwrap();
    assert!(
        !settings.is_path_read_only(&similar_name),
        "Files with 'generator' in name but not in generated dir should not be read-only"
    );
}

#[test]
fn test_is_path_read_only_with_specific_paths() {
    let settings = make_settings_with_read_only(&["vendor/**", "node_modules/**"]);

    let vendor_file = RelPath::new(Path::new("vendor/lib/package.js"), PathStyle::local()).unwrap();
    assert!(
        settings.is_path_read_only(&vendor_file),
        "Files in vendor directory should be read-only"
    );

    let node_modules_file = RelPath::new(
        Path::new("node_modules/lodash/index.js"),
        PathStyle::local(),
    )
    .unwrap();
    assert!(
        settings.is_path_read_only(&node_modules_file),
        "Files in node_modules should be read-only"
    );

    let src_file = RelPath::new(Path::new("src/app.js"), PathStyle::local()).unwrap();
    assert!(
        !settings.is_path_read_only(&src_file),
        "Files in src should not be read-only"
    );
}

#[test]
fn test_is_path_read_only_empty_patterns() {
    let settings = make_settings_with_read_only(&[]);

    let any_file = RelPath::new(Path::new("src/main.rs"), PathStyle::local()).unwrap();
    assert!(
        !settings.is_path_read_only(&any_file),
        "No files should be read-only when patterns are empty"
    );
}

#[test]
fn test_is_path_read_only_with_extension_pattern() {
    let settings = make_settings_with_read_only(&["**/*.lock", "**/*.min.js"]);

    let lock_file = RelPath::new(Path::new("Cargo.lock"), PathStyle::local()).unwrap();
    assert!(
        settings.is_path_read_only(&lock_file),
        "Lock files should be read-only"
    );

    let nested_lock =
        RelPath::new(Path::new("packages/app/yarn.lock"), PathStyle::local()).unwrap();
    assert!(
        settings.is_path_read_only(&nested_lock),
        "Nested lock files should be read-only"
    );

    let minified_js = RelPath::new(Path::new("dist/bundle.min.js"), PathStyle::local()).unwrap();
    assert!(
        settings.is_path_read_only(&minified_js),
        "Minified JS files should be read-only"
    );

    let regular_js = RelPath::new(Path::new("src/app.js"), PathStyle::local()).unwrap();
    assert!(
        !settings.is_path_read_only(&regular_js),
        "Regular JS files should not be read-only"
    );
}

#[gpui::test]
fn test_glob_settings_preserve_private_siblings(cx: &mut gpui::TestAppContext) {
    cx.update(|cx| {
        let mut store = settings::SettingsStore::new(cx, &settings::default_settings());
        store.register_setting::<WorktreeSettings>();
        store.set_user_settings(r#"{"private_files": ["**/*.key", "["], "file_scan_exclusions": ["**/.git", "target/**", "["], "read_only_files": ["**/*.gen.rs", "["]}"#, cx).expect("valid JSON settings");
        let settings = store.get::<WorktreeSettings>(None);
        for path in [".env.local", "keys/signing.key"] {assert!(settings.is_path_private(util::rel_path::rel_path(path)), "private protection lost: {path}");}
        assert!(!settings.is_path_private(util::rel_path::rel_path("src/main.rs")));
        assert!(settings.is_path_excluded(util::rel_path::rel_path("nested/.git/config")));
        assert!(settings.is_path_excluded(util::rel_path::rel_path("target/file.rs")));
        assert!(!settings.is_path_excluded(util::rel_path::rel_path("targeted/file.rs")));
        assert!(settings.is_path_read_only(util::rel_path::rel_path("src/types.gen.rs")));
        assert!(!settings.is_path_read_only(util::rel_path::rel_path("src/main.rs")));
    });
}

#[gpui::test]
fn test_glob_settings_preserve_inclusion_ancestors(cx: &mut gpui::TestAppContext) {
    cx.update(|cx| {
        let mut store = settings::SettingsStore::new(cx, &settings::default_settings());
        store.register_setting::<WorktreeSettings>();
        store
            .set_user_settings(
                r#"{"file_scan_inclusions": ["{foo/bar,baz/qux}/file.rs", "["]}"#,
                cx,
            )
            .expect("valid JSON settings");
        let settings = store.get::<WorktreeSettings>(None);
        for path in ["foo", "foo/bar", "baz", "baz/qux"] {
            assert!(
                settings.is_path_always_included(util::rel_path::rel_path(path), true),
                "required inclusion ancestor missing: {path}"
            );
        }
        assert!(
            settings.is_path_always_included(util::rel_path::rel_path("foo/bar/file.rs"), false)
        );
        assert!(
            settings.is_path_always_included(util::rel_path::rel_path("baz/qux/file.rs"), false)
        );
        assert!(!settings.is_path_always_included(util::rel_path::rel_path("other"), true));
    });
}

#[gpui::test]
fn test_glob_settings_resource_failure_and_reload(cx: &mut gpui::TestAppContext) {
    cx.update(|cx| {
        let mut store = settings::SettingsStore::new(cx, &settings::default_settings());
        store.register_setting::<WorktreeSettings>();
        let oversized = format!("scope/{}?.k?y", "[ab]".repeat(262144));
        store.set_user_settings(&serde_json::json!({"private_files":["**/*.key",oversized],"file_scan_exclusions":["target/**",oversized],"read_only_files":["**/*.gen.rs",oversized],"hidden_files":["**/*.hidden",oversized],"file_scan_inclusions":["included/file.rs",oversized]}).to_string(), cx).expect("settings parse");
        let settings = store.get::<WorktreeSettings>(None);
        for matcher in [&settings.private_files, &settings.file_scan_exclusions, &settings.read_only_files] {assert!(matcher.is_conservative());assert!(matcher.is_match(util::rel_path::rel_path("unrelated/file.txt")));}
        assert!(settings.is_std_path_read_only(Path::new("/outside/file.txt")));
        assert!(!settings.hidden_files.is_conservative());
        assert!(!settings.file_scan_inclusions.is_conservative());
        assert!(!settings.is_path_hidden(util::rel_path::rel_path("unrelated/file.txt")));
        assert!(!settings.is_path_always_included(util::rel_path::rel_path("unrelated"), true));
        assert!(settings.is_path_always_included(util::rel_path::rel_path("included"), true));
        assert!(settings.is_path_hidden(util::rel_path::rel_path("file.hidden")));
        let old_private = settings.private_files.clone();
        store.set_user_settings(r#"{"private_files":["**/*.key"],"file_scan_exclusions":["target/**"],"read_only_files":["**/*.gen.rs"],"hidden_files":[],"file_scan_inclusions":[]}"#, cx).expect("correction parses");
        let settings = store.get::<WorktreeSettings>(None);
        assert_ne!(old_private, settings.private_files);
        for matcher in [&settings.private_files, &settings.file_scan_exclusions, &settings.read_only_files] {assert!(!matcher.is_conservative());assert!(!matcher.is_match(util::rel_path::rel_path("unrelated/file.txt")));}
        assert!(settings.is_path_private(util::rel_path::rel_path(".env.local")));
        assert!(settings.is_path_private(util::rel_path::rel_path("keys/signing.key")));
        assert!(settings.is_path_excluded(util::rel_path::rel_path("target/file.rs")));
        assert!(!settings.is_std_path_read_only(Path::new("/outside/file.txt")));
        assert!(settings.is_path_read_only(util::rel_path::rel_path("src/types.gen.rs")));
    });
}

#[gpui::test]
fn test_glob_settings_aggregate_recovery(cx: &mut gpui::TestAppContext) {
    cx.update(|cx| {
        let mut store = settings::SettingsStore::new(cx, &settings::default_settings());
        store.register_setting::<WorktreeSettings>();
        let patterns = (0..512)
            .map(|index| format!("scope{index}/{}?.k?y", "[ab]".repeat(512)))
            .collect::<Vec<_>>();
        store
            .set_user_settings(
                &serde_json::json!({"private_files":patterns,"file_scan_inclusions":patterns})
                    .to_string(),
                cx,
            )
            .expect("settings parse");
        let settings = store.get::<WorktreeSettings>(None);
        let path = format!("scope511/{}x.key", "a".repeat(512));
        assert!(!settings.private_files.is_conservative());
        assert!(settings.is_path_private(util::rel_path::rel_path(&path)));
        assert!(settings.is_path_always_included(util::rel_path::rel_path(&path), false));
        assert!(settings.is_path_always_included(util::rel_path::rel_path("scope511"), true));
        assert!(!settings.is_path_private(util::rel_path::rel_path("outside/file.txt")));
        assert!(!settings.is_path_always_included(util::rel_path::rel_path("outside"), true));
    });
}
fn settings_with_patterns(setting: &str, patterns: &[&str]) -> WorktreeSettings {
    let mut content = SettingsContent::parse_json_with_comments(&settings::default_settings())
        .expect("default settings");
    let inherited: SettingsContent =
        serde_json::from_value(json!({ setting: ["**/.git"] })).expect("inherited settings");
    content.merge_from(&inherited);
    let overrides: SettingsContent =
        serde_json::from_value(json!({ setting: patterns })).expect("override settings");
    content.merge_from(&overrides);
    WorktreeSettings::from_settings(&content)
}

#[test]
fn test_invalid_direct_inclusions_do_not_include_parents() {
    let settings =
        settings_with_patterns("file_scan_inclusions", &["ignored/[", "valid/nested/**"]);
    assert!(!settings.is_path_always_included(rel_path("ignored"), true));
    assert!(!settings.is_path_always_included(rel_path("ignored/file.txt"), false));
    assert!(settings.is_path_always_included(rel_path("valid"), true));
    assert!(settings.is_path_always_included(rel_path("valid/nested"), true));
    assert!(settings.is_path_always_included(rel_path("valid/nested/file.txt"), false));
}

#[test]
fn test_inclusions_preserve_brace_patterns() {
    for (pattern, files) in [
        (
            "ignored/{one/two,three}/file.rs",
            ["ignored/one/two/file.rs", "ignored/three/file.rs"],
        ),
        (
            "{one/two,three}/file.rs",
            ["one/two/file.rs", "three/file.rs"],
        ),
    ] {
        assert!(PathMatcher::new([pattern], PathStyle::local()).is_ok());
        let settings =
            settings_with_patterns("file_scan_inclusions", &[pattern, "valid/nested/**"]);
        for file in files {
            assert!(settings.is_path_always_included(rel_path(file), false));
            for parent in rel_path(file)
                .ancestors()
                .skip(1)
                .filter(|path| !path.is_empty())
            {
                assert!(
                    settings.is_path_always_included(parent, true),
                    "{pattern}: missing ancestor {parent:?}"
                );
            }
        }
        assert!(settings.is_path_always_included(rel_path("valid"), true));
        assert!(settings.is_path_always_included(rel_path("valid/nested/file.txt"), false));
        assert!(!settings.is_path_always_included(rel_path("unmatched/file.txt"), false));
        assert!(
            !settings.is_path_always_included(rel_path("unmatched"), true),
            "{pattern}: unexpected ancestor"
        );
    }
}

#[test]
fn test_inclusion_parents_close_nested_braces() {
    let settings = settings_with_patterns("file_scan_inclusions", &["{a/{b,c},d}/file.rs"]);
    for directory in ["a", "a/b", "a/c", "d"] {
        assert!(
            settings.is_path_always_included(rel_path(directory), true),
            "missing ancestor {directory}"
        );
    }
    for directory in ["b", "c", "unmatched"] {
        assert!(
            !settings.is_path_always_included(rel_path(directory), true),
            "unexpected ancestor {directory}"
        );
    }
    assert!(settings.is_path_always_included(rel_path("a/b/file.rs"), false));
    assert!(settings.is_path_always_included(rel_path("d/file.rs"), false));
}

#[test]
fn test_inclusion_parents_close_braces_around_character_classes() {
    let settings = settings_with_patterns("file_scan_inclusions", &["{a,[x/y]z}/file.rs"]);
    assert!(settings.is_path_always_included(rel_path("a"), true));
    assert!(settings.is_path_always_included(rel_path("xz"), true));
    assert!(settings.is_path_always_included(rel_path("a/file.rs"), false));
    assert!(settings.is_path_always_included(rel_path("xz/file.rs"), false));
}

#[test]
fn test_inclusion_parents_preserve_literal_braces() {
    let mut patterns = vec!["ignored/[{]one/file.rs"];
    if PathStyle::local().is_posix() {
        patterns.push(r"ignored/\{one/file.rs");
    }
    for pattern in patterns {
        let settings = settings_with_patterns("file_scan_inclusions", &[pattern]);
        assert!(settings.is_path_always_included(rel_path("ignored/{one/file.rs"), false));
        assert!(settings.is_path_always_included(rel_path("ignored/{one"), true));
        assert!(settings.is_path_always_included(rel_path("ignored"), true));
        assert!(!settings.is_path_always_included(rel_path("ignored/other/file.rs"), false));
    }
}

#[cfg(not(windows))]
#[test]
fn test_inclusion_parents_preserve_escaping() {
    let settings = settings_with_patterns("file_scan_inclusions", &[r"{one\/two,three}/file.rs"]);
    assert!(settings.is_path_always_included(rel_path("one/two/file.rs"), false));
    assert!(settings.is_path_always_included(rel_path("one/two"), true));
    assert!(settings.is_path_always_included(rel_path("one"), true));

    let settings = settings_with_patterns("file_scan_inclusions", &[r"ignored\/nested/file.rs"]);
    assert!(settings.is_path_always_included(rel_path("ignored/nested/file.rs"), false));
    assert!(settings.is_path_always_included(rel_path("ignored/nested"), true));
    assert!(settings.is_path_always_included(rel_path("ignored"), true));
}

#[cfg(not(windows))]
#[test]
fn test_inclusion_parents_preserve_character_classes() {
    let settings = settings_with_patterns("file_scan_inclusions", &[r"[a\]cache/file.rs"]);
    assert!(settings.is_path_always_included(rel_path("acache/file.rs"), false));
    assert!(settings.is_path_always_included(rel_path("acache"), true));
    assert!(!settings.is_path_always_included(rel_path("unmatched"), true));
}

#[test]
fn test_private_files_preserves_valid_patterns() {
    let settings = settings_with_patterns("private_files", &["**/private/**"]);
    assert!(settings.is_path_private(rel_path("private/file.txt")));
    assert!(settings.is_path_private(rel_path(".env.local")));

    let settings = settings_with_patterns("private_files", &["**/private/**", "["]);
    assert!(settings.is_path_private(rel_path("private/file.txt")));
    assert!(settings.is_path_private(rel_path(".env.local")));
    assert!(!settings.is_path_private(rel_path("src/main.rs")));
    assert!(PathMatcher::new(["**/private/**", "["], PathStyle::local()).is_err());

    for patterns in [&["["][..], &[][..]] {
        let settings = settings_with_patterns("private_files", patterns);
        assert!(settings.is_path_private(rel_path(".env.local")));
        assert!(!settings.is_path_private(rel_path("private/file.txt")));
    }
}

#[test]
fn test_glob_settings_preserve_current_vec_replacement() {
    let cases: [(&str, fn(&WorktreeSettings) -> &PathMatcher); 4] = [
        ("file_scan_exclusions", |settings| {
            &settings.file_scan_exclusions
        }),
        ("file_scan_inclusions", |settings| {
            &settings.file_scan_inclusions
        }),
        ("hidden_files", |settings| &settings.hidden_files),
        ("read_only_files", |settings| &settings.read_only_files),
    ];
    for (setting, matcher) in cases {
        let settings = settings_with_patterns(setting, &["**/alpha/**", "[", "zeta/**"]);
        assert!(matcher(&settings).is_match(rel_path("alpha/file.txt")));
        assert!(matcher(&settings).is_match(rel_path("zeta/file.txt")));
        assert!(!matcher(&settings).is_match(rel_path(".git")));
        assert!(!matcher(&settings).is_match(rel_path("unmatched/file.txt")));
        for patterns in [vec![], vec!["[", "{"]] {
            let settings = settings_with_patterns(setting, &patterns);
            assert!(!matcher(&settings).is_match(rel_path(".git")));
            assert!(!matcher(&settings).is_match(rel_path("alpha/file.txt")));
        }
        let settings = settings_with_patterns(setting, &["..."]);
        assert!(!matcher(&settings).is_match(rel_path(".git")));
        assert!(matcher(&settings).is_match(rel_path("...")));
    }
}
