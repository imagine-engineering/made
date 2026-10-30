use agent::buzz_tool_compat::{
    BuzzToolCompatibilityError, BuzzToolCompatibilityMapper, BuzzToolRequest,
    NativeBuzzToolRequest, bound_buzz_tool_output,
};
use serde_json::{Value, json};

fn mapper() -> BuzzToolCompatibilityMapper {
    BuzzToolCompatibilityMapper::new("workspace").expect("valid default worktree")
}

fn map(name: &str, arguments: Value) -> NativeBuzzToolRequest {
    mapper()
        .map(
            BuzzToolRequest {
                name: name.to_owned(),
                arguments,
            },
            false,
            |_| true,
        )
        .expect("compatibility mapping succeeds")
}

fn tool(request: NativeBuzzToolRequest) -> agent::buzz_tool_compat::NativeBuzzToolCall {
    match request {
        NativeBuzzToolRequest::Tool(call) => call,
        NativeBuzzToolRequest::CurrentPlan | NativeBuzzToolRequest::ReplacePlan(_) => {
            panic!("expected native tool call")
        }
    }
}

#[test]
fn maps_every_supported_buzz_tool_to_a_native_owner() {
    let shell = tool(map(
        "shell",
        json!({"command":"cargo check", "workdir":"workspace/crate", "timeout_ms":900000}),
    ));
    assert_eq!(shell.tool_name, "terminal");
    assert_eq!(shell.arguments["cd"], "workspace/crate");
    assert_eq!(shell.arguments["timeout_ms"], 600_000);

    let read = tool(map(
        "read_file",
        json!({"path":"src/main.rs", "offset":9, "limit":20}),
    ));
    assert_eq!(read.tool_name, "read_file");
    assert_eq!(read.arguments["path"], "workspace/src/main.rs");
    assert_eq!(read.arguments["start_line"], 10);
    assert_eq!(read.arguments["end_line"], 29);

    let edit = tool(map(
        "str_replace",
        json!({"path":"src/main.rs", "old_str":"before", "new_str":"after"}),
    ));
    assert_eq!(edit.tool_name, "edit_file");
    assert_eq!(edit.arguments["edits"][0]["old_text"], "before");
    assert_eq!(edit.arguments["edits"][0]["new_text"], "after");

    let search = tool(map(
        "rg",
        json!({"pattern":"SessionId", "path":"src", "glob":"**/*.rs", "case_sensitive":true}),
    ));
    assert_eq!(search.tool_name, "grep");
    assert_eq!(search.arguments["regex"], "SessionId");
    assert_eq!(search.arguments["include_pattern"], "workspace/src/**/*.rs");
    assert_eq!(search.arguments["case_sensitive"], true);

    let scoped_search = tool(map(
        "search",
        json!({"regex":"token", "workdir":"workspace/crate"}),
    ));
    assert_eq!(
        scoped_search.arguments["include_pattern"],
        "workspace/crate/**/*"
    );

    let tree = tool(map("tree", json!({"path":"src", "depth":1})));
    assert_eq!(tree.tool_name, "list_directory");
    assert_eq!(tree.arguments["path"], "workspace/src");

    let image = tool(map(
        "view_image",
        json!({"source":"assets/icon.png", "max_dim":1568}),
    ));
    assert_eq!(image.tool_name, "read_file");
    assert_eq!(image.arguments["path"], "workspace/assets/icon.png");

    assert!(matches!(
        map("todo", json!({})),
        NativeBuzzToolRequest::CurrentPlan
    ));
    let NativeBuzzToolRequest::ReplacePlan(plan) = map(
        "todo",
        json!({"todos":[{"text":" first ","done":false},{"text":"second","done":true}]}),
    ) else {
        panic!("expected native plan replacement");
    };
    assert_eq!(plan.entries.len(), 2);
    assert_eq!(plan.entries[0].content, "first");
    assert_eq!(
        plan.entries[1].status,
        agent_client_protocol::schema::v1::PlanEntryStatus::Completed
    );
}

#[test]
fn native_availability_denial_fails_before_dispatch() {
    let result = mapper().map(
        BuzzToolRequest {
            name: "shell".to_owned(),
            arguments: json!({"command":"git status"}),
        },
        false,
        |tool_name| tool_name != "terminal",
    );
    assert_eq!(result, Err(BuzzToolCompatibilityError::Denied));
}

#[test]
fn invalid_and_remote_paths_fail_closed() {
    for (name, arguments) in [
        ("read_file", json!({"path":"../secret"})),
        (
            "str_replace",
            json!({"path":"/tmp/file", "old_str":"a", "new_str":"b"}),
        ),
        ("tree", json!({"path":"src/../../secret"})),
        (
            "view_image",
            json!({"source":"https://example.com/private.png"}),
        ),
    ] {
        assert_eq!(
            mapper().map(
                BuzzToolRequest {
                    name: name.to_owned(),
                    arguments,
                },
                false,
                |_| true,
            ),
            Err(BuzzToolCompatibilityError::InvalidPath),
        );
    }
}

#[test]
fn unsupported_legacy_semantics_are_not_silently_changed() {
    for (name, arguments, expected) in [
        (
            "read_file",
            json!({"path":"src/main.rs", "limit":0}),
            BuzzToolCompatibilityError::InvalidArguments,
        ),
        (
            "tree",
            json!({"path":"src", "depth":3}),
            BuzzToolCompatibilityError::InvalidArguments,
        ),
        (
            "str_replace",
            json!({"path":"src/main.rs", "old_str":"a", "new_str":"b", "replace_all":true}),
            BuzzToolCompatibilityError::ReplaceAllUnsupported,
        ),
        (
            "todo",
            json!({"todos":[{"text":"duplicate"},{"text":" duplicate "}]}),
            BuzzToolCompatibilityError::InvalidArguments,
        ),
        (
            "todo",
            json!({"todos":[{"text":"spoof\u{202e}txt"}]}),
            BuzzToolCompatibilityError::InvalidArguments,
        ),
    ] {
        assert_eq!(
            mapper().map(
                BuzzToolRequest {
                    name: name.to_owned(),
                    arguments,
                },
                false,
                |_| true,
            ),
            Err(expected),
        );
    }
}

#[test]
fn debug_output_redacts_tool_arguments_and_plan_contents() {
    let shell = map("shell", json!({"command":"printenv SECRET_TOKEN"}));
    let plan = map(
        "todo",
        json!({"todos":[{"text":"rotate SECRET_TOKEN", "done":false}]}),
    );

    assert!(!format!("{shell:?}").contains("SECRET_TOKEN"));
    assert!(!format!("{plan:?}").contains("SECRET_TOKEN"));
}

#[test]
fn output_is_tail_bounded_on_utf8_boundaries() {
    let output = format!("private-prefix\n{}", "🦀".repeat(3_000));
    let bounded = bound_buzz_tool_output(&output);
    assert!(bounded.starts_with("[output truncated]\n"));
    assert!(!bounded.contains("private-prefix"));
    assert!(bounded.len() <= 8 * 1_024 + "[output truncated]\n".len());
}

#[test]
fn cancellation_prevents_mapping_or_native_dispatch() {
    let mut availability_called = false;
    let result = mapper().map(
        BuzzToolRequest {
            name: "shell".to_owned(),
            arguments: json!({"command":"sleep 30"}),
        },
        true,
        |_| {
            availability_called = true;
            true
        },
    );
    assert_eq!(result, Err(BuzzToolCompatibilityError::Cancelled));
    assert!(!availability_called);
}
